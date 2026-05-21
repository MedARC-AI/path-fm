#!/usr/bin/env python3
"""Frozen TSS probe for OpenMidnight checkpoints.

This diagnostic asks whether tissue-source-site (TSS) information is linearly
recoverable from frozen CLS embeddings. It samples TCGA tiles from the existing
training manifest, extracts frozen features, and trains a scikit-learn
LogisticRegression probe with WSI-level train/test splitting to avoid leakage.

Typical cluster usage:

    uv run python tools/tss_probe.py \
      --config-file dinov2/configs/train/vitg14_reg4.yaml \
      --sample-list-path /block/TCGA/sample_dataset_30.txt \
      --checkpoint-path output_full_adv_lam50/eval/training_10000/teacher_checkpoint.pth \
      --output-json output_full_adv_lam50/tss_probe_training_10000.json

For the Meta DINOv2-G baseline, omit --checkpoint-path and pass:

    --torchhub-model dinov2_vitg14_reg
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Iterable

@dataclass(frozen=True)
class TileRecord:
    path: str
    x: int
    y: int
    level: int
    wsi_id: str
    tss: str


@dataclass(frozen=True)
class WsiSplit:
    train_wsi_ids: list[str]
    test_wsi_ids: list[str]
    train_labels: list[str]
    test_labels: list[str]
    class_names: list[str]


@dataclass(frozen=True)
class ProbeRecords:
    records: list[TileRecord]
    labels: list[str]
    wsi_ids: list[str]


def parse_sample_list_line(line: str) -> TileRecord:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Expected '<path> <x> <y> <level>', got: {line!r}")

    path = parts[0]
    basename = os.path.basename(path)
    name_parts = basename.split("-")
    if len(name_parts) < 2 or name_parts[0] != "TCGA":
        raise ValueError(f"Cannot extract TSS from non-TCGA filename: {basename}")

    return TileRecord(
        path=path,
        x=int(parts[1]),
        y=int(parts[2]),
        level=int(parts[3]),
        wsi_id=basename,
        tss=name_parts[1],
    )


def load_manifest_records(sample_list_path: str) -> tuple[dict[str, list[TileRecord]], dict[str, set[str]]]:
    records_by_wsi: dict[str, list[TileRecord]] = defaultdict(list)
    wsi_by_tss: dict[str, set[str]] = defaultdict(set)
    with open(sample_list_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = parse_sample_list_line(line)
            records_by_wsi[record.wsi_id].append(record)
            wsi_by_tss[record.tss].add(record.wsi_id)
    return dict(records_by_wsi), dict(wsi_by_tss)


def split_wsi_ids_by_tss(
    wsi_by_tss: dict[str, set[str]],
    max_wsis_per_class: int,
    test_size: float,
    seed: int,
) -> WsiSplit:
    """Select a bounded WSI set per TSS and split at WSI level.

    The split is done independently per class so rare-but-valid TSS classes
    still get at least one train and one test WSI. A global stratified split can
    fail when the number of TSS classes is large relative to the sample count.
    """
    if max_wsis_per_class < 2:
        raise ValueError("max_wsis_per_class must be at least 2")

    rng = random.Random(seed)
    train_wsi_ids: list[str] = []
    test_wsi_ids: list[str] = []
    train_labels: list[str] = []
    test_labels: list[str] = []
    class_names: list[str] = []

    for tss in sorted(wsi_by_tss):
        wsi_ids = sorted(wsi_by_tss[tss])
        if len(wsi_ids) < 2:
            continue
        rng.shuffle(wsi_ids)
        wsi_ids = wsi_ids[:max_wsis_per_class]
        if len(wsi_ids) < 2:
            continue
        n_test = min(len(wsi_ids) - 1, max(1, round(len(wsi_ids) * test_size)))
        test_for_class = wsi_ids[:n_test]
        train_for_class = wsi_ids[n_test:]
        test_wsi_ids.extend(test_for_class)
        train_wsi_ids.extend(train_for_class)
        test_labels.extend([tss] * len(test_for_class))
        train_labels.extend([tss] * len(train_for_class))
        class_names.append(tss)

    if len(class_names) < 2:
        raise ValueError("Need at least two TSS classes with >=2 WSIs")

    return WsiSplit(
        train_wsi_ids=train_wsi_ids,
        test_wsi_ids=test_wsi_ids,
        train_labels=train_labels,
        test_labels=test_labels,
        class_names=class_names,
    )


def select_probe_records(
    records_by_wsi: dict[str, list[TileRecord]],
    wsi_ids: Iterable[str],
    max_tiles_per_wsi: int,
    seed: int,
) -> ProbeRecords:
    rng = random.Random(seed)
    records: list[TileRecord] = []
    labels: list[str] = []
    selected_wsi_ids: list[str] = []

    for wsi_id in sorted(wsi_ids):
        candidates = list(records_by_wsi[wsi_id])
        rng.shuffle(candidates)
        for record in candidates[:max_tiles_per_wsi]:
            records.append(record)
            labels.append(record.tss)
            selected_wsi_ids.append(wsi_id)

    return ProbeRecords(records=records, labels=labels, wsi_ids=selected_wsi_ids)


def find_latest_teacher_checkpoint(output_dir: str) -> str:
    """Find the highest-step eval teacher checkpoint in an OpenMidnight output dir."""
    eval_dir = Path(output_dir) / "eval"
    candidates: list[tuple[int, Path]] = []
    if not eval_dir.exists():
        raise FileNotFoundError(f"No eval directory found under {output_dir}")

    for path in eval_dir.glob("training_*/teacher_checkpoint.pth"):
        match = re.search(r"training_(\d+)$", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            f"No eval/training_*/teacher_checkpoint.pth files found under {output_dir}. "
            "Pass --checkpoint-path explicitly, or run once after an eval checkpoint exists."
        )
    return str(max(candidates, key=lambda item: item[0])[1])


def _load_cfg(config_file: str):
    from omegaconf import OmegaConf

    from dinov2.configs import dinov2_default_config

    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(config_file)
    return OmegaConf.merge(default_cfg, cfg)


def _filter_backbone_state(state_dict: dict) -> dict:
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    prefixes = (
        "teacher.backbone.",
        "teacher.module.backbone.",
        "backbone.",
    )
    for prefix in prefixes:
        filtered = {
            k.removeprefix(prefix): v
            for k, v in cleaned.items()
            if k.startswith(prefix)
        }
        if filtered:
            return filtered
    return cleaned


def load_model(config_file: str, checkpoint_path: str | None, torchhub_model: str | None, device: str):
    import torch

    if checkpoint_path and torchhub_model:
        raise ValueError("Use either --checkpoint-path or --torchhub-model, not both")

    if torchhub_model:
        model = torch.hub.load("facebookresearch/dinov2", torchhub_model, pretrained=True)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model

    from dinov2.models import build_model_from_cfg

    cfg = _load_cfg(config_file)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        load_msg = model.load_state_dict(_filter_backbone_state(state_dict), strict=False)
        print(f"Loaded checkpoint with msg: {load_msg}")
    else:
        raise ValueError("Provide --checkpoint-path or --torchhub-model")

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _preprocess_batch(records: list[TileRecord], patch_size: int):
    import torch
    from openslide import OpenSlide
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensors = []
    slide_cache: dict[str, OpenSlide] = {}
    try:
        for record in records:
            slide = slide_cache.get(record.path)
            if slide is None:
                slide = OpenSlide(record.path)
                slide_cache[record.path] = slide
            patch = slide.read_region((record.x, record.y), level=record.level, size=(patch_size, patch_size))
            tensors.append(transform(patch.convert("RGB")))
    finally:
        for slide in slide_cache.values():
            slide.close()
    return torch.stack(tensors, dim=0)


def extract_cls_embeddings(
    model,
    records: list[TileRecord],
    batch_size: int,
    patch_size: int,
    device: str,
) -> np.ndarray:
    import numpy as np
    import torch

    features = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = _preprocess_batch(batch_records, patch_size).to(device, non_blocking=True)
            if hasattr(model, "forward_features"):
                output = model.forward_features(images)
                cls = output["x_norm_clstoken"]
            else:
                cls = model(images)
            features.append(cls.float().cpu().numpy())
            print(f"Extracted {min(start + batch_size, len(records))}/{len(records)} tiles", flush=True)
    return np.concatenate(features, axis=0)


def fit_logistic_probe(
    x_train: np.ndarray,
    y_train: list[str],
    x_test: np.ndarray,
    y_test: list[str],
    max_iter: int,
    seed: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            multi_class="auto",
            n_jobs=-1,
            random_state=seed,
            solver="saga",
        ),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="dinov2/configs/train/vitg14_reg4.yaml")
    parser.add_argument("--sample-list-path", default="/block/TCGA/sample_dataset_30.txt")
    parser.add_argument("--checkpoint-path", default=None, help="teacher_checkpoint.pth or compatible training checkpoint")
    parser.add_argument("--checkpoint-dir", default=None, help="OpenMidnight output dir; uses latest eval/training_*/teacher_checkpoint.pth")
    parser.add_argument("--torchhub-model", default=None, help="Example: dinov2_vitg14_reg for Meta baseline")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-wsis-per-class", type=int, default=10)
    parser.add_argument("--max-tiles-per-wsi", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if args.checkpoint_dir:
        if checkpoint_path:
            raise ValueError("Use either --checkpoint-path or --checkpoint-dir, not both")
        checkpoint_path = find_latest_teacher_checkpoint(args.checkpoint_dir)
        print(f"Resolved latest teacher checkpoint: {checkpoint_path}")

    records_by_wsi, wsi_by_tss = load_manifest_records(args.sample_list_path)
    split = split_wsi_ids_by_tss(
        wsi_by_tss,
        max_wsis_per_class=args.max_wsis_per_class,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_records = select_probe_records(records_by_wsi, split.train_wsi_ids, args.max_tiles_per_wsi, args.seed)
    test_records = select_probe_records(records_by_wsi, split.test_wsi_ids, args.max_tiles_per_wsi, args.seed + 1)

    print(
        f"Probe classes={len(split.class_names)} "
        f"train_tiles={len(train_records.records)} test_tiles={len(test_records.records)} "
        f"train_wsis={len(split.train_wsi_ids)} test_wsis={len(split.test_wsi_ids)}"
    )

    model = load_model(args.config_file, checkpoint_path, args.torchhub_model, args.device)
    x_train = extract_cls_embeddings(model, train_records.records, args.batch_size, args.patch_size, args.device)
    x_test = extract_cls_embeddings(model, test_records.records, args.batch_size, args.patch_size, args.device)
    metrics = fit_logistic_probe(x_train, train_records.labels, x_test, test_records.labels, args.max_iter, args.seed)

    result = {
        "checkpoint_path": checkpoint_path,
        "torchhub_model": args.torchhub_model,
        "sample_list_path": args.sample_list_path,
        "n_classes": len(split.class_names),
        "class_names": split.class_names,
        "train_tiles": len(train_records.records),
        "test_tiles": len(test_records.records),
        "train_wsis": len(split.train_wsi_ids),
        "test_wsis": len(split.test_wsi_ids),
        "max_wsis_per_class": args.max_wsis_per_class,
        "max_tiles_per_wsi": args.max_tiles_per_wsi,
        "test_size": args.test_size,
        "seed": args.seed,
        "metrics": metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

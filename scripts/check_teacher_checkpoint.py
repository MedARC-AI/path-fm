# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Load a teacher_checkpoint.pth for inference via setup_and_build_model and
report any missing / unexpected keys, then run a forward sanity pass.

Example:
    uv run python scripts/check_teacher_checkpoint.py \
        --run /data/vbelagali/vits_run1 \
        --iteration 137500

    # or point at the files directly:
    uv run python scripts/check_teacher_checkpoint.py \
        --config-file /data/vbelagali/vits_run1/config.yaml \
        --pretrained-weights /data/vbelagali/vits_run1/eval/training_137500/teacher_checkpoint.pth

Needs a GPU (build_model_for_eval calls model.cuda()), e.g. via:
    srun --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:10:00 --pty \
        bash -lc 'export PYTHONPATH=$PWD; uv run python scripts/check_teacher_checkpoint.py --run ...'
"""

import argparse
import logging
import os
import tempfile

import torch

from dinov2.eval.setup import setup_and_build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None,
                   help="Run output dir; derives --config-file and --pretrained-weights with --iteration")
    p.add_argument("--iteration", type=str, default=None,
                   help="Training iteration to load when --run is given (e.g. 137500)")
    p.add_argument("--config-file", type=str, default=None, help="Path to the run's config.yaml")
    p.add_argument("--pretrained-weights", type=str, default=None, help="Path to teacher_checkpoint.pth")
    p.add_argument("--image-size", type=int, default=224, help="Square input size for the forward sanity pass")
    return p.parse_args()


def resolve_paths(a):
    config_file = a.config_file
    weights = a.pretrained_weights
    if a.run is not None:
        if config_file is None:
            config_file = os.path.join(a.run, "config.yaml")
        if weights is None:
            if a.iteration is None:
                raise SystemExit("--iteration is required when using --run without --pretrained-weights")
            weights = os.path.join(a.run, "eval", f"training_{a.iteration}", "teacher_checkpoint.pth")
    if config_file is None or weights is None:
        raise SystemExit("Provide either --run (+--iteration) or both --config-file and --pretrained-weights")
    return config_file, weights


def main():
    a = parse_args()
    config_file, weights = resolve_paths(a)

    # Surface dinov2's INFO logs, including the load_state_dict msg from
    # load_pretrained_weights (missing_keys / unexpected_keys).
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    # setup() writes config.yaml into output_dir; use a throwaway dir so we
    # never clobber the real run's config.
    tmp = tempfile.mkdtemp(prefix="ckpt_load_test_")
    args = argparse.Namespace(
        config_file=config_file,
        pretrained_weights=weights,
        output_dir=tmp,
        opts=[],
    )

    print("=== running setup_and_build_model end-to-end ===")
    print(f"config : {config_file}")
    print(f"weights: {weights}")

    model, autocast_dtype = setup_and_build_model(args)

    print("\n=== RESULT ===")
    print("autocast_dtype :", autocast_dtype)
    print("model class    :", type(model).__name__)
    print("device         :", next(model.parameters()).device)
    print("training mode  :", model.training)
    print("param count (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 3))

    # Independently recompute missing / unexpected against the eval model,
    # replicating the exact key transforms in load_pretrained_weights.
    sd = torch.load(weights, map_location="cpu", weights_only=False)["teacher"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)

    print("\n=== explicit missing / unexpected (re-loaded) ===")
    print("missing_keys    :", len(res.missing_keys))
    for k in res.missing_keys:
        print("   MISSING   ", k)
    print("unexpected_keys :", len(res.unexpected_keys))
    for k in res.unexpected_keys:
        print("   UNEXPECTED", k)

    # Sanity forward pass.
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
        x = torch.randn(2, 3, a.image_size, a.image_size, device="cuda")
        out = model(x)
    print("\n=== forward sanity ===")
    print("input :", tuple(x.shape))
    print("output:", tuple(out.shape), out.dtype)


if __name__ == "__main__":
    main()

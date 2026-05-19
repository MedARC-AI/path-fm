"""Tests for _build_tss_mapping and the TSS extraction logic in train.py."""
import os
import textwrap

import pytest
import torch

from dinov2.train.train import _build_tss_mapping


# ── TSS code extraction from TCGA filenames ───────────────────────────────────

@pytest.mark.parametrize("path,expected_tss", [
    ("/data/TCGA/TCGA-A1-A0SO-01Z-00-DX1.svs", "A1"),
    ("/data/TCGA/TCGA-BH-A0BM-01A-01-BSA.svs", "BH"),
    ("/data/TCGA/TCGA-E2-A14X-01Z-00-DX1.svs", "E2"),
    ("TCGA-XX-YYYY-01.svs", "XX"),
])
def test_tss_extraction_from_tcga_filename(path, expected_tss):
    tss = os.path.basename(path).split("-")[1]
    assert tss == expected_tss


# ── _build_tss_mapping ────────────────────────────────────────────────────────

def _write_sample_list(tmp_path, lines):
    p = tmp_path / "sample_list.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _cfg(sample_list_path, min_wsi_count):
    from omegaconf import OmegaConf
    return OmegaConf.create(
        {
            "train": {"sample_list_path": sample_list_path},
            "adversarial": {"min_wsi_count": min_wsi_count},
        }
    )


def test_tss_codes_are_sorted(tmp_path):
    lines = [
        "/data/TCGA-ZZ-001.svs tile1",
        "/data/TCGA-AA-001.svs tile2",
        "/data/TCGA-MM-001.svs tile3",
        "/data/TCGA-ZZ-002.svs tile4",  # second WSI for ZZ
        "/data/TCGA-AA-002.svs tile5",  # second WSI for AA
        "/data/TCGA-MM-002.svs tile6",  # second WSI for MM
    ]
    p = _write_sample_list(tmp_path, lines)
    codes, _ = _build_tss_mapping(_cfg(p, min_wsi_count=2))
    assert codes == sorted(codes)
    assert codes == ["AA", "MM", "ZZ"]


def test_min_wsi_threshold_filters_tss(tmp_path):
    """TSS codes with fewer than min_wsi_count unique WSI paths are excluded."""
    lines = [
        # AA: 3 unique WSIs — included at threshold 3
        "/data/TCGA-AA-001.svs x",
        "/data/TCGA-AA-001.svs y",  # duplicate path → same WSI
        "/data/TCGA-AA-002.svs z",
        "/data/TCGA-AA-003.svs w",
        # BB: 1 unique WSI — excluded
        "/data/TCGA-BB-001.svs x",
        "/data/TCGA-BB-001.svs y",
    ]
    p = _write_sample_list(tmp_path, lines)
    codes, _ = _build_tss_mapping(_cfg(p, min_wsi_count=3))
    assert "AA" in codes
    assert "BB" not in codes


def test_class_weights_are_inverse_tile_frequency(tmp_path):
    """class_weight[i] = (1/tile_count[i]) / mean(1/tile_count)."""
    lines = [
        # AA: 2 tiles, 2 WSIs
        "/data/TCGA-AA-001.svs",
        "/data/TCGA-AA-002.svs",
        # BB: 4 tiles, 2 WSIs
        "/data/TCGA-BB-001.svs",
        "/data/TCGA-BB-001.svs",  # duplicate line → same tile counted
        "/data/TCGA-BB-002.svs",
        "/data/TCGA-BB-002.svs",
    ]
    p = _write_sample_list(tmp_path, lines)
    codes, weights = _build_tss_mapping(_cfg(p, min_wsi_count=2))

    assert codes == ["AA", "BB"]
    raw = torch.tensor([1.0 / 2, 1.0 / 4])
    expected = raw / raw.mean()
    assert torch.allclose(weights, expected, atol=1e-6)


def test_class_weights_mean_normalized(tmp_path):
    """Normalising by mean ensures weights.mean() == 1.0."""
    lines = [
        "/data/TCGA-AA-001.svs",
        "/data/TCGA-AA-002.svs",
        "/data/TCGA-BB-001.svs",
        "/data/TCGA-BB-002.svs",
    ]
    p = _write_sample_list(tmp_path, lines)
    _, weights = _build_tss_mapping(_cfg(p, min_wsi_count=2))
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)


def test_empty_lines_ignored(tmp_path):
    """Blank lines in the sample list must not raise or produce extra codes."""
    lines = [
        "",
        "/data/TCGA-AA-001.svs",
        "/data/TCGA-AA-002.svs",
        "",
        "",
    ]
    p = _write_sample_list(tmp_path, lines)
    codes, weights = _build_tss_mapping(_cfg(p, min_wsi_count=2))
    assert codes == ["AA"]
    assert weights.shape == (1,)


def test_returns_tensor_weights(tmp_path):
    lines = [
        "/data/TCGA-AA-001.svs",
        "/data/TCGA-AA-002.svs",
    ]
    p = _write_sample_list(tmp_path, lines)
    _, weights = _build_tss_mapping(_cfg(p, min_wsi_count=2))
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32

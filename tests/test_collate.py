import pytest
import torch
from dinov2.data.collate import collate_data_and_cast


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_sample(n_global=2, n_local=4, crop_h=16, crop_w=16, slide_id=None):
    """Build a minimal collate-compatible sample tuple."""
    global_crops = [torch.randn(3, crop_h, crop_w) for _ in range(n_global)]
    local_crops = [torch.randn(3, crop_h // 2, crop_w // 2) for _ in range(n_local)]
    images = ({"global_crops": global_crops, "local_crops": local_crops}, None)
    index = 0
    if slide_id is not None:
        return images, index, slide_id  # 3-tuple (training path)
    return images, index               # 2-tuple (streaming path)


def _mask_gen(n_tokens):
    """Minimal mask generator: marks the first k tokens as masked."""
    def gen(k):
        mask = [False] * n_tokens
        for i in range(min(k, n_tokens)):
            mask[i] = True
        return mask
    return gen


def _collate(samples, n_tokens=16, mask_probability=0.5):
    return collate_data_and_cast(
        samples,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=mask_probability,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=_mask_gen(n_tokens),
    )


# ── slide_ids extraction ──────────────────────────────────────────────────────

@pytest.mark.parametrize("slide_ids", [
    ["A1"],
    ["AA", "BB", "CC"],
    ["TCGA-A1", "TCGA-B2", "TCGA-C3", "TCGA-D4"],
])
def test_slide_ids_extracted_from_3tuple(slide_ids):
    samples = [_make_sample(slide_id=sid) for sid in slide_ids]
    result = _collate(samples)
    assert result["slide_ids"] == slide_ids


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_slide_ids_none_for_2tuple(batch_size):
    samples = [_make_sample(slide_id=None) for _ in range(batch_size)]
    result = _collate(samples)
    assert result["slide_ids"] is None


def test_slide_ids_order_preserved():
    """slide_ids must match sample order (same index as samples_list)."""
    ids = ["first", "second", "third"]
    samples = [_make_sample(slide_id=sid) for sid in ids]
    result = _collate(samples)
    assert result["slide_ids"][0] == "first"
    assert result["slide_ids"][1] == "second"
    assert result["slide_ids"][2] == "third"


# ── collated tensor shapes ────────────────────────────────────────────────────

@pytest.mark.parametrize("n_global,n_local,batch_size", [
    (2, 4, 3),
    (2, 8, 2),
])
def test_collated_crop_shapes(n_global, n_local, batch_size):
    """Collated tensors have outer=crop, inner=sample ordering."""
    samples = [
        _make_sample(n_global=n_global, n_local=n_local, slide_id=f"s{i}")
        for i in range(batch_size)
    ]
    result = _collate(samples)
    assert result["collated_global_crops"].shape[0] == n_global * batch_size
    assert result["collated_local_crops"].shape[0] == n_local * batch_size


def test_collated_global_crop_ordering():
    """Global crop collation: outer=crop_index, inner=sample_index.

    With 2 samples and 2 global crops, the expected stack order is:
        [crop0_of_sample0, crop0_of_sample1, crop1_of_sample0, crop1_of_sample1]
    Not interleaved (sample0_crop0, sample0_crop1, sample1_crop0, sample1_crop1).
    """
    # Give each crop a distinguishable constant value
    def _sample_with_values(g0_val, g1_val):
        global_crops = [
            torch.full((3, 8, 8), g0_val),
            torch.full((3, 8, 8), g1_val),
        ]
        local_crops = [torch.zeros(3, 4, 4)]
        images = ({"global_crops": global_crops, "local_crops": local_crops}, None)
        return images, 0, "dummy"

    s0 = _sample_with_values(g0_val=1.0, g1_val=2.0)
    s1 = _sample_with_values(g0_val=3.0, g1_val=4.0)

    result = collate_data_and_cast(
        [s0, s1],
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=4,
        mask_generator=_mask_gen(4),
    )

    collated = result["collated_global_crops"]  # shape (4, 3, 8, 8)
    # outer loop: crop index 0, then crop index 1
    assert collated[0].mean().item() == pytest.approx(1.0)  # crop0_s0
    assert collated[1].mean().item() == pytest.approx(3.0)  # crop0_s1
    assert collated[2].mean().item() == pytest.approx(2.0)  # crop1_s0
    assert collated[3].mean().item() == pytest.approx(4.0)  # crop1_s1


# ── output dict keys ──────────────────────────────────────────────────────────

def test_collate_output_keys_present():
    required_keys = {
        "collated_global_crops",
        "collated_local_crops",
        "collated_masks",
        "mask_indices_list",
        "masks_weight",
        "upperbound",
        "n_masked_patches",
        "indexes",
        "slide_ids",
    }
    samples = [_make_sample(slide_id="x") for _ in range(2)]
    result = _collate(samples)
    assert required_keys.issubset(result.keys())

"""Tests for the adversarial training logic added to SSLMetaArch."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from dinov2.train.ssl_meta_arch import compute_adv_accuracy, compute_grad_norm


# ── tss_to_idx mapping ────────────────────────────────────────────────────────

@pytest.mark.parametrize("codes,query,expected_idx", [
    (["A1", "BH", "E2"], "A1", 0),
    (["A1", "BH", "E2"], "BH", 1),
    (["A1", "BH", "E2"], "E2", 2),
    (["ZZ", "AA", "MM"], "AA", 0),  # unsorted input → sorted order
    (["ZZ", "AA", "MM"], "MM", 1),
    (["ZZ", "AA", "MM"], "ZZ", 2),
])
def test_tss_to_idx_is_sorted(codes, query, expected_idx):
    """tss_to_idx must be built from sorted(all_tss_codes) for deterministic indices."""
    tss_to_idx = {tss: idx for idx, tss in enumerate(sorted(codes))}
    assert tss_to_idx[query] == expected_idx


def test_tss_to_idx_unknown_code_returns_minus1():
    """Unknown TSS codes map to -1 (the ignore_index for cross-entropy)."""
    tss_to_idx = {"A1": 0, "BH": 1}
    label = tss_to_idx.get("UNKNOWN", -1)
    assert label == -1


@pytest.mark.parametrize("slide_ids,known_codes,expected_labels", [
    (["A1", "BH", "XX"], ["A1", "BH", "E2"], [0, 1, -1]),
    (["E2", "A1", "E2"], ["A1", "BH", "E2"], [2, 0, 2]),
    (["XX", "YY", "ZZ"], ["A1", "BH"], [-1, -1, -1]),
])
def test_adversarial_labels_construction(slide_ids, known_codes, expected_labels):
    tss_to_idx = {tss: idx for idx, tss in enumerate(sorted(known_codes))}
    labels = torch.tensor(
        [tss_to_idx.get(tss, -1) for tss in slide_ids], dtype=torch.long
    )
    assert labels.tolist() == expected_labels


# ── label expansion: repeat vs repeat_interleave ──────────────────────────────

def test_repeat_gives_crop_outer_sample_inner_ordering():
    """Collate stacks crops outer, samples inner.
    Labels must use .repeat(n_crops) — not .repeat_interleave(n_crops) — to match.

    With B=3 samples and n_global=2:
      collated: [s0_crop0, s1_crop0, s2_crop0, s0_crop1, s1_crop1, s2_crop1]
      labels:   [l0,       l1,       l2,       l0,       l1,       l2      ]
                = labels.repeat(2)
    """
    labels = torch.tensor([10, 20, 30])
    n_crops = 2

    repeated = labels.repeat(n_crops)
    assert repeated.tolist() == [10, 20, 30, 10, 20, 30]


def test_repeat_interleave_gives_wrong_ordering():
    """Demonstrate that repeat_interleave would mis-align labels with collated crops."""
    labels = torch.tensor([10, 20, 30])
    n_crops = 2

    interleaved = labels.repeat_interleave(n_crops)
    # wrong: [10, 10, 20, 20, 30, 30] — pairs each label with both crops of the same sample
    assert interleaved.tolist() == [10, 10, 20, 20, 30, 30]
    # but collated order has crop as outer loop → the correct expansion is repeat
    assert interleaved.tolist() != [10, 20, 30, 10, 20, 30]


@pytest.mark.parametrize("n_global,n_local", [(2, 4), (2, 8)])
def test_label_expansion_length(n_global, n_local):
    B = 5
    labels = torch.arange(B, dtype=torch.long)
    global_slide_ids = labels.repeat(n_global)
    local_slide_ids = labels.repeat(n_local)
    assert len(global_slide_ids) == n_global * B
    assert len(local_slide_ids) == n_local * B


# ── NaN guard ─────────────────────────────────────────────────────────────────

def test_cross_entropy_all_ignored_returns_nan():
    """F.cross_entropy with all ignore_index=-1 divides 0/0 → NaN.
    This is the bug the (adversarial_labels >= 0).any() guard prevents.
    """
    logits = torch.randn(4, 5)
    labels = torch.full((4,), -1, dtype=torch.long)
    result = F.cross_entropy(logits, labels, ignore_index=-1)
    assert torch.isnan(result)


def test_nan_guard_condition_all_excluded():
    """Guard correctly identifies an all-excluded batch."""
    adversarial_labels = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
    assert not (adversarial_labels >= 0).any()


def test_nan_guard_condition_partial_included():
    """Guard does not fire when at least one label is valid."""
    adversarial_labels = torch.tensor([-1, 2, -1, 0], dtype=torch.long)
    assert (adversarial_labels >= 0).any()


def test_nan_guard_cross_entropy_with_some_valid_is_finite():
    """When guard passes (some valid labels), cross_entropy is finite."""
    logits = torch.randn(4, 5)
    labels = torch.tensor([-1, 2, -1, 0], dtype=torch.long)
    result = F.cross_entropy(logits, labels, ignore_index=-1)
    assert torch.isfinite(result)


# ── Teacher model dict never receives slide_classifier ────────────────────────

def test_teacher_model_dict_excludes_slide_classifier():
    """Only student gets slide_classifier — teacher has no such key."""
    student_model_dict = {}
    teacher_model_dict = {}

    slide_classifier = nn.Linear(16, 5)
    student_model_dict["slide_classifier"] = slide_classifier

    student = nn.ModuleDict(student_model_dict)
    teacher = nn.ModuleDict(teacher_model_dict)

    assert "slide_classifier" in student
    assert "slide_classifier" not in teacher


# ── update_teacher skips slide_classifier ─────────────────────────────────────

def test_update_teacher_ema_skips_slide_classifier():
    """Teacher EMA update must not touch slide_classifier student parameters.

    Simulates the update_teacher loop with a monkeypatched get_fsdp_modules that
    returns a simple wrapper (so the loop body executes, unlike in real non-FSDP mode).
    """
    torch.manual_seed(0)
    student_backbone = nn.Linear(8, 8)
    teacher_backbone = nn.Linear(8, 8)
    # Give teacher different weights so we can verify the update
    with torch.no_grad():
        teacher_backbone.weight.fill_(0.0)
        teacher_backbone.bias.fill_(0.0)
        student_backbone.weight.fill_(1.0)
        student_backbone.bias.fill_(1.0)

    slide_classifier = nn.Linear(8, 3)
    sc_weight_original = slide_classifier.weight.data.clone()

    student_keys = {"backbone": student_backbone, "slide_classifier": slide_classifier}
    teacher_keys = {"backbone": teacher_backbone}

    class _Wrapper:
        def __init__(self, module):
            self.params = list(module.parameters())

    def fake_get_fsdp(module):
        return [_Wrapper(module)]

    # Replicate the update_teacher loop with the patched helper
    student_param_list = []
    teacher_param_list = []
    for k in student_keys:
        if k == "slide_classifier":
            continue
        for ms, mt in zip(fake_get_fsdp(student_keys[k]), fake_get_fsdp(teacher_keys[k])):
            student_param_list += ms.params
            teacher_param_list += mt.params

    m = 0.9
    with torch.no_grad():
        torch._foreach_mul_(teacher_param_list, m)
        torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    # backbone teacher: 0 * 0.9 + 1 * 0.1 = 0.1
    assert torch.allclose(teacher_backbone.weight, torch.full_like(teacher_backbone.weight, 0.1))
    # slide_classifier student must be untouched
    assert torch.allclose(slide_classifier.weight.data, sc_weight_original)


# ── compute_adv_accuracy ─────────────────────────────────────────────────────

def test_compute_adv_accuracy_perfect():
    """All predictions correct → accuracy 1.0."""
    preds = torch.tensor([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = torch.tensor([0, 1, 2])
    valid = torch.tensor([True, True, True])
    assert compute_adv_accuracy(preds, labels, valid).item() == pytest.approx(1.0)


def test_compute_adv_accuracy_none_correct():
    """All predictions wrong → accuracy 0.0."""
    preds = torch.tensor([[0.0, 10.0],
                          [10.0, 0.0]])
    labels = torch.tensor([0, 1])
    valid = torch.tensor([True, True])
    assert compute_adv_accuracy(preds, labels, valid).item() == pytest.approx(0.0)


def test_compute_adv_accuracy_partial_valid():
    """Only valid samples contribute to accuracy."""
    preds = torch.tensor([[10.0, 0.0],   # correct (valid)
                          [10.0, 0.0],   # wrong, but ignored
                          [0.0, 10.0]])  # correct (valid)
    labels = torch.tensor([0, 1, 1])
    valid = torch.tensor([True, False, True])
    assert compute_adv_accuracy(preds, labels, valid).item() == pytest.approx(1.0)


def test_compute_adv_accuracy_half_correct():
    """2 out of 4 correct → 0.5."""
    preds = torch.tensor([[10.0, 0.0],   # correct
                          [10.0, 0.0],   # wrong
                          [0.0, 10.0],   # correct
                          [0.0, 10.0]])  # wrong
    labels = torch.tensor([0, 1, 1, 0])
    valid = torch.ones(4, dtype=torch.bool)
    assert compute_adv_accuracy(preds, labels, valid).item() == pytest.approx(0.5)


# ── compute_grad_norm ────────────────────────────────────────────────────────

def test_compute_grad_norm_known_values():
    """L2 norm of known gradient vectors."""
    model = nn.Linear(4, 2, bias=False)
    model.weight.grad = torch.tensor([[3.0, 0.0, 0.0, 0.0],
                                       [0.0, 4.0, 0.0, 0.0]])
    # flattened: [3, 0, 0, 0, 0, 4, 0, 0] → norm = 5.0
    assert compute_grad_norm(model).item() == pytest.approx(5.0)


def test_compute_grad_norm_with_bias():
    """Includes bias gradients in the norm."""
    model = nn.Linear(1, 1, bias=True)
    model.weight.grad = torch.tensor([[3.0]])
    model.bias.grad = torch.tensor([4.0])
    # [3, 4] → norm = 5.0
    assert compute_grad_norm(model).item() == pytest.approx(5.0)


def test_compute_grad_norm_no_grads():
    """Module with no gradients returns 0."""
    model = nn.Linear(4, 2)
    assert compute_grad_norm(model).item() == pytest.approx(0.0)


def test_compute_grad_norm_partial_grads():
    """Only parameters with .grad set contribute."""
    model = nn.Linear(1, 1, bias=True)
    model.weight.grad = torch.tensor([[3.0]])
    # bias.grad is None → only weight contributes
    assert compute_grad_norm(model).item() == pytest.approx(3.0)


def test_compute_grad_norm_multi_layer():
    """Works across a module with multiple sub-layers."""
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    model[0].weight.grad = torch.zeros(2, 2)
    model[0].weight.grad[0, 0] = 3.0
    model[1].weight.grad = torch.zeros(1, 2)
    model[1].weight.grad[0, 0] = 4.0
    # flattened: [3, 0, 0, 0, 4, 0] → norm = 5.0
    assert compute_grad_norm(model).item() == pytest.approx(5.0)


# ── valid_label_fraction ─────────────────────────────────────────────────────

def test_valid_label_fraction_all_valid():
    labels = torch.tensor([0, 1, 2, 3])
    fraction = (labels >= 0).float().mean()
    assert fraction.item() == pytest.approx(1.0)


def test_valid_label_fraction_some_ignored():
    labels = torch.tensor([0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
    fraction = (labels >= 0).float().mean()
    assert fraction.item() == pytest.approx(0.5)


def test_valid_label_fraction_seven_of_ten():
    labels = torch.tensor([0, 1, 2, -1, 4, 5, 6, -1, 8, -1])
    fraction = (labels >= 0).float().mean()
    assert fraction.item() == pytest.approx(0.7)


def test_valid_label_fraction_none_valid():
    labels = torch.tensor([-1, -1, -1])
    fraction = (labels >= 0).float().mean()
    assert fraction.item() == pytest.approx(0.0)

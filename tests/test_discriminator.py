import pytest
import torch
import torch.nn as nn

from dinov2.models.discriminator import GradientReversalLayer, GradientReversalFunction, SlideClassifier


# ── GradientReversalLayer ─────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_grl_forward_is_identity(alpha):
    grl = GradientReversalLayer(alpha)
    x = torch.randn(4, 8)
    out = grl(x)
    assert torch.allclose(out, x)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_grl_backward_negates_and_scales_gradient(alpha):
    """∂L/∂x = -alpha * ∂L/∂y — the core property of the gradient reversal layer."""
    grl = GradientReversalLayer(alpha)
    x = torch.randn(4, 8, requires_grad=True)
    grl(x).sum().backward()
    expected = -alpha * torch.ones_like(x)
    assert torch.allclose(x.grad, expected)


def test_grl_alpha_zero_zeros_gradient():
    """alpha=0 blocks the adversarial gradient entirely."""
    grl = GradientReversalLayer(alpha=0.0)
    x = torch.randn(4, 8, requires_grad=True)
    grl(x).sum().backward()
    assert torch.all(x.grad == 0)


def test_grl_gradient_sign_is_reversed_vs_identity():
    """GRL gradient has opposite sign compared to a plain nn.Identity."""
    alpha = 1.0
    grl = GradientReversalLayer(alpha)

    x_grl = torch.randn(4, 8, requires_grad=True)
    x_id = x_grl.detach().clone().requires_grad_(True)

    grl(x_grl).sum().backward()
    x_id.sum().backward()

    assert torch.allclose(x_grl.grad, -x_id.grad)


# ── SlideClassifier ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_dim,hidden_dim,num_classes,batch", [
    (16, 32, 5, 8),
    (64, 128, 10, 1),
    (256, 256, 20, 16),
])
def test_slide_classifier_output_shape(input_dim, hidden_dim, num_classes, batch):
    model = SlideClassifier(input_dim, hidden_dim, num_classes, alpha=1.0)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(batch, input_dim))
    assert out.shape == (batch, num_classes)


def test_slide_classifier_hidden_dims():
    """MLP halves width at each layer: input→d, d→d/2, d/2→d/4, d/4→num_classes."""
    input_dim, hidden_dim, num_classes = 64, 128, 10
    model = SlideClassifier(input_dim, hidden_dim, num_classes, alpha=1.0)
    linear_layers = [m for m in model.classifier if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 4
    assert linear_layers[0].in_features == input_dim
    assert linear_layers[0].out_features == hidden_dim           # hidden_dim1
    assert linear_layers[1].out_features == hidden_dim // 2      # hidden_dim2
    assert linear_layers[2].out_features == hidden_dim // 4      # hidden_dim3
    assert linear_layers[3].out_features == num_classes


def test_slide_classifier_gradients_flow_through_grl():
    """Gradients must reach input features (GRL does not block them, only reverses)."""
    model = SlideClassifier(input_dim=16, hidden_dim=32, num_classes=5, alpha=1.0)
    x = torch.randn(4, 16, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_slide_classifier_gradients_reversed():
    """Replacing GRL alpha with -1 (double reversal) should match a plain forward pass."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, requires_grad=False)

    # Model with alpha=1.0 (adversarial use): gradients are negated
    m_adv = SlideClassifier(16, 32, 5, alpha=1.0)
    # Model with alpha=-1.0: double reversal → same sign as plain forward
    m_double = SlideClassifier(16, 32, 5, alpha=-1.0)
    # Copy identical weights
    m_double.load_state_dict(m_adv.state_dict())

    x_adv = x.clone().requires_grad_(True)
    x_double = x.clone().requires_grad_(True)

    m_adv(x_adv).sum().backward()
    m_double(x_double).sum().backward()

    # alpha=1 negates; alpha=-1 double-negates → double should equal positive forward grad
    # i.e. x_double.grad == -x_adv.grad
    assert torch.allclose(x_double.grad, -x_adv.grad, atol=1e-5)

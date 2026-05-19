# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported verbatim from lotterlab/advdino dinov2/train/ssl_meta_arch.py lines 81-135.

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class SlideClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, alpha):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        hidden_dim1 = hidden_dim
        hidden_dim2 = hidden_dim1 // 2
        hidden_dim3 = hidden_dim2 // 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim3, num_classes),
        )

    def forward(self, features):
        return self.classifier(self.grl(features))

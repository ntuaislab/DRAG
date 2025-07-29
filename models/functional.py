"""
functional.py

Utility functions for models.
"""

from typing import Sequence

import torch
from torch import Tensor
from torch.autograd import Function
from torchvision.transforms import v2


def slerp(p0: Tensor, p1: Tensor, t: float) -> Tensor:
    """
    Perform Spherical Linear Interpolation between two 4D tensors.

    Parameters
    ----------
    t : float
        Interpolation factor (0.0 to 1.0).
        If 0, the output is p0; if 1, the output is p1.

    p0 : Tensor
        Start tensor with shape (B, C, H, W).

    p1 : Tensor
        End tensor with shape (B, C, H, W).

    Returns
    -------
    Tensor
        Interpolated tensor with shape (B, C, H, W).
    """
    # Normalize the vectors
    p0_norm = p0 / p0.flatten(1).norm(dim=1)[:, None, None, None]
    p1_norm = p1 / p1.flatten(1).norm(dim=1)[:, None, None, None]

    # Compute the cosine of the angle between the vectors
    dot_product = (p0_norm * p1_norm).sum(dim=[1, 2, 3]).clamp(-1.0, 1.0)  # Ensure within [-1, 1]

    # Compute the angle between the vectors
    # If the angle is small, use linear interpolation
    omega = torch.acos(dot_product)
    mask = (torch.abs(omega) < 0.01)[:, None, None, None]

    # Compute the spherical interpolation
    denominator = torch.sin(omega)
    coeff_1 = (torch.sin((1.0 - t) * omega) / denominator)[:, None, None, None]
    coeff_2 = (torch.sin(t * omega) / denominator)[:, None, None, None]
    interp = coeff_1 * p0 + coeff_2 * p1

    return torch.where(mask, torch.lerp(p0, p1, t), interp)


class UnNormalize(v2.Normalize):
    """
    References
    ----------
    https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False) -> None:
        super().__init__(
            [-m / s for m, s in zip(mean, std)],
            [1.0 / s for s in std],
            inplace
        )


class WeightedGradient(Function):
    @staticmethod
    def forward(ctx, input, gamma):
        ctx.gamma = gamma
        return input

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs # Unpack the tuple (assert 1 elem in the tuple)
        gamma = ctx.gamma
        return gamma * grad_output, None


def invert_gradient(input):
    return WeightedGradient.apply(input, -1)

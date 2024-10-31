from typing import Tuple

import numpy as np
import scipy.ndimage as ndi  # type: ignore
import torch

from . import random


def random_mask(shape: Tuple[int, ...] = (1024, 1024), verbose=False) -> torch.Tensor:
    """Generate a random ellipse mask roughly centered in the middle

    Args:
        shape (Tuple[int, ...]): Size of the image.
            Default: (1024, 1024)
        verbose (bool): If True print the true area versus the sampled area

    Returns:
        torch.Tensor: Boolean mask
            Shape: shape, dtype: bool

    """
    # Maths:
    # A_2d = pi * a * b
    # A_3d = 4/3 pi * a * b * c
    # b = ratio[1] * a, c = ratio[2] * a
    # let's call r the biggest possible ratio (and 1/r the smallest one)
    # Then the largest possible axe is
    # a_max_2d = (A_2d / pi * r).pow(1/2)
    # a_max_3d = (A_3d / pi * r.pow(3) * 3/4).pow(1/3)
    # We choose A and r so that a_max < 0.5
    dim = len(shape)
    assert dim in (2, 3)

    shape_pt = torch.tensor(shape)

    mask_area = torch.rand(1, generator=random.particle_generator).item() * 0.15 + 0.2  # [0.2, 0.35]
    if dim == 3:
        # The volume of the ellipse vs square is 2/3 smaller in 3D than in 2D.
        # This is still not enough to prevent the radius to be too large (see maths)
        mask_area *= 1 / 2  # [0.1, 0.17]

    ratios = torch.ones(dim)
    ratios[1:] = 0.5 + 1.5 * torch.rand(dim - 1, generator=random.particle_generator)  # [0.5, 2]
    if dim == 3:
        ratios.sqrt_()  # So that the smallest ratio possible vs the largest one is still at a ratio 2

    mean = shape_pt / 2 + torch.randn(len(shape), generator=random.particle_generator) * shape_pt / 60

    # 4/3 pi abc in 3D vs pi ab in 2D
    coef = 4 / 3 * np.pi if dim == 3 else np.pi
    std = (mask_area / coef / torch.prod(ratios)).pow(1 / dim)
    std = std * ratios
    std = std * torch.prod(torch.tensor(shape)).pow(1 / dim)

    indices = torch.tensor(np.indices(shape), dtype=torch.float32).permute(*range(1, dim + 1), 0)
    mask = ((indices - mean) / std).pow(2).sum(dim=-1) <= 1

    if verbose:
        print(mean, std)
        print(mask.sum() / mask.numel(), mask_area)

    return mask


def mask_from_frame(frame: np.ndarray) -> torch.Tensor:
    """Find the animal mask using a simple thresholding method

    We threshold at the maximum inflexion point.

    Args:
        frame (np.ndarray): Frame of the video
            Shape: ([D, ]H, W, 1), dtype: float

    Returns:
        torch.Tensor: Boolean mask
            Shape: ([D, ]H, W), dtype: bool
    """
    # First blur the image
    frame = ndi.gaussian_filter(frame[..., 0], [10.0] * (frame.ndim - 1))

    # Limit the search to threshold resulting in 10 to 40% of the image
    mini = int((np.quantile(frame, 0.6) * 100).round())
    maxi = int((np.quantile(frame, 0.9) * 100).round())

    bins = np.array([k / 100 for k in range(101)])
    hist, _ = np.histogram(frame.ravel(), bins=bins)

    # Smoothing of the histogram before inflexion extraction
    cumsum = np.cumsum(hist)
    cumsum_pad = np.pad(cumsum, 10 // 2, mode="edge")
    cumsum_smooth = np.convolve(cumsum_pad, np.ones(10) / 10, mode="valid")

    opt = np.gradient(np.gradient(cumsum_smooth))[mini : maxi + 1].argmin() + mini
    # opt = np.gradient(np.gradient(np.gradient(cumsum_smooth)))[mini : maxi + 1].argmax() + mini

    threshold = bins[opt + 1]

    return torch.tensor(frame > threshold)

import cv2
import numpy as np
import torch

from . import random


def random_mask(size=1024, verbose=False) -> torch.Tensor:
    """Generate a random ellipse mask roughly centered in the middle

    Args:
        size (int): Size of the image.
            Default: 1024
        verbose (bool): If True print the true area versus the sampled area

    Returns:
        torch.Tensor: 2D boolean mask
            Shape: (1024, 1024), dtype: bool

    """
    thresh = 1 / np.sqrt(2 * np.pi) ** 2 * np.exp(-1 / 2)  # Thresh at 1 sigma (1 mahalanohobis)
    mask_area = torch.rand(1, generator=random.particle_generator).item() * 0.15 + 0.2  # [0.2, 0.35]
    ratio = 0.5 + 1.5 * torch.rand(1, generator=random.particle_generator).item()  # [0.5, 2]

    mean = size / 2 + torch.randn(2, generator=random.particle_generator) * size / 60

    std = torch.tensor([mask_area * ratio / torch.pi, mask_area / ratio / torch.pi]).sqrt() * size

    distribution = torch.distributions.Normal(mean, std)

    indices = torch.tensor(np.indices((size, size)), dtype=torch.float32).permute(1, 2, 0)
    prob = distribution.log_prob(indices).sum(dim=2).exp() * std.prod()

    mask = prob > thresh
    if verbose:
        print(mask.sum() / mask.numel(), mask_area)

    return mask


def mask_from_frame(frame: np.ndarray) -> torch.Tensor:
    """Find the animal mask using a simple thresholding method

    We threshold at the maximum inflexion point.

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (H, W, 1), dtype: float

    Returns:
        torch.Tensor: 2D boolean mask
            Shape: (H, W), dtype: bool
    """
    # First blur the image
    frame = cv2.GaussianBlur(frame[..., 0], (35, 35), 10.0, 10.0)  # type: ignore

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

import dataclasses
from typing import Tuple

import numba  # type: ignore
import numpy as np
import scipy.ndimage as ndi  # type: ignore
import torch

from . import random


def _build_rot_2d(theta: torch.Tensor) -> torch.Tensor:
    """Build 2d rot matrices from the angles

    R = |cos -sin|
        |sin  cos|

    Args:
        theta (torch.Tensor): The rotation angles
            Shape: (n, 1)

    Returns:
        torch.Tensor: Rotation matrices (2d)
            Shape: (n, 2, 2)

    """
    theta = theta[..., 0]
    rot = torch.empty((len(theta), 2, 2), dtype=theta.dtype)
    rot[:, 0, 0] = torch.cos(theta)
    rot[:, 1, 0] = torch.sin(theta)
    rot[:, 0, 1] = -rot[:, 1, 0]
    rot[:, 1, 1] = rot[:, 0, 0]
    return rot


def _build_rot_3d(theta: torch.Tensor) -> torch.Tensor:
    """Build 3d rot matrices from the angles

    It apply a 2d rotation along each axis (j then i then k).

        |1  0    0 |   |cos  0 sin|   |cos -sin 0|
    R = |0 cos -sin| @ | 0   1  0 | @ |sin  cos 0|
        |0 sin  cos|   |-sin 0 cos|   | 0    0  1|

    Args:
        theta (torch.Tensor): The rotation angles
            Shape: (n, 3)

    Returns:
        torch.Tensor: Rotation matrices (2d)
            Shape: (n, 2, 2)

    """
    rot_k = torch.zeros(len(theta), 3, 3, dtype=theta.dtype)
    rot_k[:, 0, 0] = 1
    rot_k[:, 1:, 1:] = _build_rot_2d(theta[:, :1])

    rot_i = torch.zeros(len(theta), 3, 3, dtype=theta.dtype)
    rot_i[:, 0, 0] = 1
    rot_i[:, 1:, 1:] = _build_rot_2d(theta[:, 1:2])
    rot_i = torch.roll(rot_i, (1, 1), (1, 2))  # Roll axis to have the rot along the second axis (i)

    rot_j = torch.zeros(len(theta), 3, 3, dtype=theta.dtype)
    rot_j[:, 0, 0] = 1
    rot_j[:, 1:, 1:] = _build_rot_2d(theta[:, 2:])
    rot_j = torch.roll(rot_j, (2, 2), (1, 2))  # Roll axis to have the rot along the last axis (j)

    return rot_k @ rot_i @ rot_j


@numba.njit(parallel=True)
def _fast_mahalanobis_pdist(mu: np.ndarray, sigma_inv: np.ndarray, thresh: float) -> np.ndarray:
    """Compute the mahalanobis distance between mu[j] and N(mu[i], sigma[i]) for all i and j

    Note: The distance is not symmetric and dist[i, i] = inf for all i

    Args:
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        sigma_inv (np.ndarray): Inverse Covariance matrices (precision matrices)
            Shape: (n, d, d), dtype: float32
        thresh (float): Threshold on the l_inf distance when to skip the computation of the
            mahalanobis distance and set it to inf instead

    Returns:
        np.ndarray: Mahalanobis distance between each distribution
            Shape: (n, n)
    """
    n = mu.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for i in numba.prange(n):  # pylint: disable=not-an-iterable
        for j in numba.prange(i + 1, n):  # pylint: disable=not-an-iterable
            if np.abs(mu[i] - mu[j]).max() > thresh:
                continue
            delta_mu = mu[i] - mu[j]
            dist[i, j] = np.sqrt(delta_mu @ sigma_inv[i] @ delta_mu)
            dist[j, i] = np.sqrt(delta_mu @ sigma_inv[j] @ delta_mu)
    return dist


@numba.njit()
def _fast_valid(dist: np.ndarray, min_dist: float) -> np.ndarray:
    """Compute a validity mask

    The validity mask is a non-unique boolean mask defining a subset of particles where
    all pairwise distances are greater than min_dist

    Args:
        dist (np.ndarray): Distance between particles
            Shape: (n, n), dtype: float32
        min_dist (float): Minimum distance to keep

    Returns:
        np.ndarray: Validity mask
            Shape: (N,), dtype: bool
    """
    n = dist.shape[0]
    valid = np.full((n,), True)

    for i in range(n):
        if not valid[i]:
            continue
        for j in range(n):
            if not valid[j]:
                continue
            if dist[i, j] < min_dist:
                valid[j] = False

    return valid


@numba.njit
def _fast_draw_2d(samples: np.ndarray, size: Tuple[int, int], weights: np.ndarray) -> np.ndarray:
    """Generate a black image where the samples are added with their weights

    Args:
        samples (np.ndarray): Samples for each particles.
            Shape (n, m, 2), dtype: long
        size (Tuple[int, int]): Size of the image generated
        weights (np.ndarray): Weight for each particle

    Returns:
        np.ndarray: The generated image
    """
    image = np.zeros(size, dtype=np.float32)

    for particle in range(len(samples)):  # pylint: disable=consider-using-enumerate
        for sample in samples[particle]:
            if sample.min() < 0 or sample[0] >= size[0] or sample[1] >= size[1]:
                continue

            image[sample[0], sample[1]] += weights[particle]

    return image


@numba.njit
def _fast_draw_3d(samples: np.ndarray, size: Tuple[int, int, int], weights: np.ndarray) -> np.ndarray:
    """Generate a black image where the samples are added with their weights

    Args:
        samples (np.ndarray): Samples for each particles.
            Shape (n, m, 3), dtype: long
        size (Tuple[int, ...]): Size of the image generated
        weights (np.ndarray): Weight for each particle

    Returns:
        np.ndarray: The generated image
    """
    image = np.zeros(size, dtype=np.float32)

    for particle in range(len(samples)):  # pylint: disable=consider-using-enumerate
        for sample in samples[particle]:
            if sample.min() < 0 or sample[0] >= size[0] or sample[1] >= size[1]:
                continue

            image[sample[0], sample[1], sample[2]] += weights[particle]

    return image


def torch_gmm_pdf(
    indices: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor, weight: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """Fast computation of the GMM pdf given mu, sigma_inv and weights with pytorch

    We do not compute a true pdf, but rather a weighted sum of gaussian pdfs
    (scaled so that the max is at one for each gaussian no matter the covariance)

    All implementations found are using too much memory for our use case (they keep for each indice and
    each component the prob) moreover they compute the true pdf which is more expensive.

    This exploits gpu/cpu parallelization and computes the pdf only inside a valid gate based on l1 distance.
    It directly sums the pdf of each gaussian, without storing them in ram.

    Nonetheless, it computes the l2 norm between each pixel and each gaussian.

    Args:
        indices (torch.Tensor): Indices where to compute the pdf
            Shape: (m, d), dtype: float32
        mu (torch.Tensor): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        precision (torch.Tensor): Inverse covariance of each normal distribution
            Shape: (n, d, d), dtype: float32
        weight (torch.Tensor): Weight of each gaussian components
            Shape: (n,), dtype: float32
        thresholds (torch.Tensor): Threshold on the l2 distance when to skip computation
            Shape: (n,), dtype: float32

    Returns:
        torch.Tensor: Gaussian pdf for the given indices
            Shape: (n,), dtype: float32
    """
    # Move to GPU if possible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    indices = indices.to(device)
    mu = mu.to(device)
    precision = precision.to(device)
    weight = weight.to(device)
    thresholds = thresholds.to(device) ** 2

    n = mu.shape[0]
    m = indices.shape[0]
    pdf = torch.zeros((m,), dtype=torch.float32, device=device)
    for i in range(n):
        delta = indices - mu[i]
        valid = delta.pow(2).sum(dim=-1) < thresholds[i]
        delta = delta[valid]
        pdf[valid] += weight[i] * torch.exp(-0.5 * delta[:, None] @ precision[i] @ delta[..., None])[:, 0, 0]
    return pdf


@numba.njit(parallel=True)
def numba_gmm_pdf_2d(  # pylint: disable=too-many-locals
    shape: Tuple[int, int],
    mu: np.ndarray,
    precision: np.ndarray,
    weight: np.ndarray,
    thresholds: np.ndarray,
    *,
    scale=1,
) -> np.ndarray:
    """Fast computation of the GMM pdf given mu, sigma_inv and weights

    We do not compute a true pdf, but rather a weighted sum of gaussian pdfs
    (scaled so that the max is at one for each gaussian no matter the covariance)

    All implementations found are using too much memory for our use case (they keep for each indice and
    each component the prob) moreover they compute the true pdf which is more expensive.

    This exploits cpu parallelization and computes the pdf only inside a valid gate. It allows not compute
    the l2 norm for each pixel and each gaussian, reducing computationnal cost when particles are smalls.
    But it only works for 2D images, by computing the value for every pixel (scaled).

    Args:
        shape (Tuple[int, int]): Target output shape (H, W)
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        precision (np.ndarray): Inverse covariance of each normal distribution
            Shape: (n, d, d), dtype: float32
        weight (np.ndarray): Weight of each gaussian components
            Shape: (n,), dtype: float32
        thresholds (np.ndarray): Threshold on the l2 distance when to skip computation
            Shape: (n,), dtype: float32
        scale (int): Scaling to reduce computations

    Returns:
        np.ndarray: Gaussian pdf for the scaled pixels
            Shape: (H // scale, W // scale), dtype: float32
    """
    height, width = shape
    height = height // scale
    width = width // scale

    thresh = round(thresholds.max()) // scale
    box = np.indices((thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 0) - thresh
    box = box.reshape(-1, 2)

    mu_round = np.round(mu / scale).astype(np.int64)
    n = mu.shape[0]
    m = box.shape[0]

    thresholds = thresholds**2  # Squared norm vs squared thresholds

    pdf = np.zeros((height, width), dtype=np.float32)
    for k in range(n):
        for l in numba.prange(m):  # pylint: disable=not-an-iterable
            pos = box[l] + mu_round[k]

            i, j = pos

            if not 0 <= i < height or not 0 <= j < width:
                continue

            delta = pos * scale - mu[k]

            if delta @ delta >= thresholds[k]:
                continue

            maha = delta.reshape(1, 2) @ precision[k] @ delta.reshape(2, 1)
            pdf[i, j] += weight[k] * np.exp(-0.5 * maha)[0, 0]

    return pdf


@numba.njit(parallel=True)
def numba_gmm_pdf_3d(  # pylint: disable=too-many-locals
    shape: Tuple[int, int, int],
    mu: np.ndarray,
    precision: np.ndarray,
    weight: np.ndarray,
    thresholds: np.ndarray,
    *,
    scale=1,
) -> np.ndarray:
    """3d version of the numba_gmm_pdf_2d

    Args:
        shape (Tuple[int, int, int]): Target output shape (D, H, W)
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        precision (np.ndarray): Inverse covariance of each normal distribution
            Shape: (n, d, d), dtype: float32
        weight (np.ndarray): Weight of each gaussian components
            Shape: (n,), dtype: float32
        thresholds (np.ndarray): Threshold on the l2 distance when to skip computation
            Shape: (n,), dtype: float32
        scale (int): Scaling to reduce computations

    Returns:
        np.ndarray: Gaussian pdf for the scaled pixels
            Shape: (D // scale, H // scale, W // scale), dtype: float32
    """
    depth, height, width = shape
    depth = depth // scale
    height = height // scale
    width = width // scale

    thresh = round(thresholds.max()) // scale
    box = np.indices((thresh * 2 + 1, thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 3, 0) - thresh
    box = box.reshape(-1, 3)

    mu_round = np.round(mu / scale).astype(np.int64)
    n = mu.shape[0]
    m = box.shape[0]

    thresholds = thresholds**2  # Squared norm vs squared thresholds

    pdf = np.zeros((depth, height, width), dtype=np.float32)
    for k in range(n):
        for l in numba.prange(m):  # pylint: disable=not-an-iterable
            pos = box[l] + mu_round[k]

            z, i, j = pos

            if not 0 <= z < depth or not 0 <= i < height or not 0 <= j < width:
                continue

            delta = pos * scale - mu[k]

            if delta @ delta >= thresholds[k]:
                continue

            maha = delta.reshape(1, 3) @ precision[k] @ delta.reshape(3, 1)
            pdf[z, i, j] += weight[k] * np.exp(-0.5 * maha)[0, 0]

    return pdf


@numba.njit(parallel=True)
def _fast_segmentation_2d(  # pylint: disable=too-many-locals
    shape: Tuple[int, int],
    mu: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
    min_pdf: float,
) -> np.ndarray:
    """Fast computation of the segmentation mask of the particles.

    Very similar to the numba_gmm_pdf.

    Args:
        shape (Tuple[int, int]): Target output shape (H, W)
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        precision (np.ndarray): Inverse covariance of each normal distribution
            Shape: (n, d, d), dtype: float32
        thresholds (np.ndarray): Threshold on the l1 distance when to skip computation
            Shape: (n,), dtype: float32
        min_pdf (float): Minimum value of the normalized pdf to belong to the segmentation mask.

    Returns:
        np.ndarray: Gaussian pdf for the scaled pixels
            Shape: (H, W), dtype: int32
    """
    thresh = round(thresholds.max())
    box = np.indices((thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 0) - thresh
    box = box.reshape(-1, 2)

    mu_round = np.round(mu).astype(np.int64)
    n = mu.shape[0]
    m = box.shape[0]

    thresholds = thresholds**2  # Squared norm vs squared thresholds
    max_maha = -2 * np.log(min_pdf)  # Compare with maha and not pdf

    segmentation = np.zeros(shape, dtype=np.int32)
    for k in range(n):
        for l in numba.prange(m):  # pylint: disable=not-an-iterable
            pos = box[l] + mu_round[k]

            i, j = pos

            if not 0 <= i < shape[0] or not 0 <= j < shape[1]:
                continue

            delta = pos - mu[k]

            if delta @ delta >= thresholds[k]:
                continue

            maha = delta.reshape(1, 2) @ precision[k] @ delta.reshape(2, 1)

            if maha[0, 0] < max_maha:  # No need to compute exp(-0.5 maha)
                segmentation[i, j] = k + 1

    return segmentation


@numba.njit(parallel=True)
def _fast_segmentation_3d(  # pylint: disable=too-many-locals
    shape: Tuple[int, int, int],
    mu: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
    min_pdf: float,
) -> np.ndarray:
    """3D version of _fast_segmentation_2d

    Args:
        shape (Tuple[int, int, int]): Target output shape (D, H, W)
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        precision (np.ndarray): Inverse covariance of each normal distribution
            Shape: (n, d, d), dtype: float32
        thresholds (np.ndarray): Threshold on the l1 distance when to skip computation
            Shape: (n,), dtype: float32
        min_pdf (float): Minimum value of the normalized pdf to belong to the segmentation mask.

    Returns:
        np.ndarray: Gaussian pdf for the scaled pixels
            Shape: (D, H, W), dtype: int32
    """
    thresh = round(thresholds.max())
    box = np.indices((thresh * 2 + 1, thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 3, 0) - thresh
    box = box.reshape(-1, 3)

    mu_round = np.round(mu).astype(np.int64)
    n = mu.shape[0]
    m = box.shape[0]

    thresholds = thresholds**2  # Squared norm vs squared thresholds
    max_maha = -2 * np.log(min_pdf)  # Compare with maha and not pdf

    segmentation = np.zeros(shape, dtype=np.int32)
    for k in range(n):
        for l in numba.prange(m):  # pylint: disable=not-an-iterable
            pos = box[l] + mu_round[k]

            z, i, j = pos

            if not 0 <= z < shape[0] or not 0 <= i < shape[1] or not 0 <= j < shape[2]:
                continue

            delta = pos - mu[k]

            if delta @ delta >= thresholds[k]:
                continue

            maha = delta.reshape(1, 3) @ precision[k] @ delta.reshape(3, 1)

            if maha[0, 0] < max_maha:  # No need to compute exp(-0.5 maha)
                segmentation[z, i, j] = k + 1

    return segmentation


@dataclasses.dataclass
class GaussianParticlesConfig:
    """Gaussian particles configuations

    Attributes:
        n (int): Number of particles to generate. Note that due to randomness (resp. the suppression of
            too close particles), the true number of particles generated is not exactly (resp. lower than) n
        min_std, max_std (float): Minimum/Maximum std. Stds are generated uniformly between these values
        min_dist (float): Minimum mahalanobis distance between two particles. If two particles are sampled too close
            one of the two is filtered. It prevents overlapping.
            Default 0.0 (Do not filter particles)
    """

    n: int
    min_std: float
    max_std: float
    min_dist: float = 0


class GaussianParticles:
    """Handle multiple gaussian particles

    Each particle is defined by a position, a deviation, an angle and an intensity (mu, std, theta, weight).

    mu is the 2d center of the particle.
    std is standart deviation of the two axes of the ellipse.
    theta is the rotation angle from the horizontal axis.
    weight is the intensity of each spot. (1 is bright, 0 is black)

    Attributes:
        size (Tuple[int, ...]): Size of the image generated
        mu (torch.Tensor): Positions of the spots (pixels)
            Shape: (n, d), dtype: float32
        std (torch.Tensor): Uncorrelated stds (pixels)
            Shape: (n, d), dtype: float32
        theta (torch.Tensor): Rotation of each spot (radian)
            In 2d, d' = 1 (a single rotation), in 3d, d' = 3 (a rotation along each axis)
            Shape: (n, d'), dtype: float32
        weight (torch.Tensor): Weight of each spot (proportional to intensity)
            Shape: (n,), dtype: float32
    """

    def __init__(self, n: int, mask: torch.Tensor, min_std: float, max_std: float):
        """Constructor

        Args:
            n (int): Number of particles to generate. Note that due to random masking, the true number
                of particles generated is not exactly n (and is stored in self._n)
            mask (torch.Tensor): Boolean mask where to generate particles in the image
                self.size is extracted from it.
                Shape: (H, W), dtype: bool
            min_std, max_std (float): Minimum/Maximum std. Stds are generated uniformly between these values

        """
        self.size = tuple(mask.shape)
        self.dim = len(self.size)
        assert self.dim in (2, 3)

        mask_proportion = mask.sum() / mask.numel()
        self.mu = torch.rand(
            int(n / mask_proportion.item()), self.dim, generator=random.particle_generator
        ) * torch.tensor(self.size)
        if self.dim == 2:
            self.mu = self.mu[mask[self.mu[:, 0].long(), self.mu[:, 1].long()]]
        else:
            self.mu = self.mu[mask[self.mu[:, 0].long(), self.mu[:, 1].long(), self.mu[:, 2].long()]]

        self._n = self.mu.shape[0]

        self.weight = torch.ones(self._n)
        self.std = min_std + torch.rand(self._n, self.dim, generator=random.particle_generator) * (max_std - min_std)
        self.theta = torch.rand(self._n, 1 if self.dim == 2 else 3, generator=random.particle_generator) * torch.pi

        self.build_distribution()

    def filter_close_particles(self, min_dist: float) -> None:
        """Drop too close particles based on mahalanobis distance

        Args:
            min_dist (float): Minimum mahalanobis distance between two particles
        """
        dist = _fast_mahalanobis_pdist(
            self.mu.numpy(),
            self._distribution.precision_matrix.mT.numpy(),  # precision.T = precision and is contiguous
            min_dist * self.std.max().item() * 2,
        )
        valid = _fast_valid(dist, min_dist)

        self.mu = self.mu[valid]

        self._n = self.mu.shape[0]
        self.weight = self.weight[valid]
        self.std = self.std[valid]
        self.theta = self.theta[valid]

        self.build_distribution()
        print(f"Filtered particles from {valid.shape[0]} to {self._n}")

    def build_distribution(self):
        """Rebuild the distributions

        To be called each time a modification is made to mu, std or theta
        """
        rot = _build_rot_2d(self.theta) if self.dim == 2 else _build_rot_3d(self.theta)

        sigma = torch.zeros((self._n, self.dim, self.dim), dtype=self.std.dtype)
        sigma[:, 0, 0] = self.std[:, 0].pow(2)
        sigma[:, 1, 1] = self.std[:, 1].pow(2)
        if self.dim == 3:
            sigma[:, 2, 2] = self.std[:, 2].pow(2)

        sigma = rot @ sigma @ rot.permute(0, 2, 1)
        sigma = (sigma + sigma.permute(0, 2, 1)) / 2  # prevent some floating error leading to non inversible matrix

        self._distribution = torch.distributions.MultivariateNormal(self.mu, sigma)
        # NOTE: A gmm could be created with torch.distributions.MixtureSameFamily(
        #   torch.distributions.Categorical(self.weight * self.std.prod(dim=-1)),
        #   self._distribution
        # )
        # This is not faster nor easier to use
        # torch.distributions.MixtureSameFamily(
        #   torch.distributions.Categorical(self.weight * self.std.prod(dim=-1)),
        #   self._distribution
        # )

    def draw_sample(self, n=20000, blur=0.0) -> torch.Tensor:
        """Draw an image of the particles

        The generation starts from a black image where we add at each sample location the weights of its particle.
        A blurring process can be added to smooth the results (With smaller n).

        Args:
            n (int): Number of samples by particles
            blur (float): std of the blurring process
                Default: 0.0 (No blurring)

        Returns:
            torch.Tensor: Image of the particles
                Shape: ([D, ]H, W), dtype: float32
        """
        # NOTE: We did not find a way to sample from a multivariate normal distribution with a generator
        samples = self._distribution.sample((n,))  # type: ignore
        samples = samples.permute(1, 0, 2).round().long()  # Shape: self._n, n, 2
        weight = self.weight * self.std.prod(dim=-1)  # By default, smaller gaussian spots have higher intensities.
        if self.dim == 2:
            image = _fast_draw_2d(samples.numpy(), self.size, weight.numpy())
        else:
            image = _fast_draw_3d(samples.numpy(), self.size, weight.numpy())
        if blur > 0:
            image = ndi.gaussian_filter(image, [blur] * self.dim)
        return torch.tensor(image) / n

    def draw_poisson(self, dt=100.0, scale=1, k=3.0, mode="auto") -> torch.Tensor:
        """Draw from ground truth with Poisson Shot noise

        Args:
            dt (float): Integration interval
                Default: 100.0
            scale (int): Down scaling to compute true pdf
                Default: 1
            k (float): Cut computations at k stds when computing the true pdf.
                Default: 3.0
            mode (str): Computations mode for the true pdf.
                "auto": Choose automatically between "torch" or "numba"
                "torch": Use pytorch implem (fast when a gpu is available)
                "numba": Use numba implem (fast when particles are small before the number of pixels)
                Default: "auto"

        Returns:
            torch.Tensor: Image of the particles
                Shape: (H, W), dtype: float32
        """

        return torch.poisson(dt * self.draw_truth(scale, k, mode), generator=random.imaging_generator) / dt

    def draw_truth(self, scale=1, k=3.0, mode="auto") -> torch.Tensor:
        """Draw the ground truth image, where the intensities are the pdf of the mixture of gaussians

        I(x) = \\sum_i w_i N(x; \\mu_i, \\Sigma_i)

        Args:
            scale (int): Downscale the image to make the computation faster (The output is interpolated)
                Default: 1
            k (float): Cut computations at k stds. (Used to compute a threshold on the l2 distance to prevent
                useless computations)
                Default: 3.0 (Neglect pixels in the border of the particles with values < exp(-1/2 3**2) ~ 0.01)
            mode (str): Computations mode.
                "auto": Choose automatically between "torch" or "numba"
                "torch": Use pytorch implem (fast when a gpu is available)
                "numba": Use numba implem (fast when particles are small before the number of pixels)
                Default: "auto"

        Returns:
            torch.Tensor: Pdf of the particles
                Shape: (H, W), dtype: float32
        """
        # Draw using pytorch for large particles before the total number of pixels.
        if mode == "torch" or mode == "auto" and (2 * k * self.std.max()) ** self.dim > np.prod(self.size) * 0.01:
            indices = torch.tensor(np.indices(tuple(size // scale for size in self.size), dtype=np.float32))
            indices = indices.permute(*range(1, self.dim + 1), 0) * scale

            truth = torch_gmm_pdf(
                indices.reshape(-1, self.dim),
                self.mu,
                self._distribution.precision_matrix.mT,  # precision.T = precision and is contiguous
                self.weight,
                self.std.max(dim=-1).values * k,
            )

            return torch.nn.functional.interpolate(
                truth.reshape((1, 1, *(size // scale for size in self.size))),
                size=self.size,
                mode="bilinear" if self.dim == 2 else "trilinear",
            )[0, 0].cpu()

        # Numba mode
        if self.dim == 2:
            truth = torch.tensor(
                numba_gmm_pdf_2d(
                    self.size,
                    self.mu.numpy(),
                    self._distribution.precision_matrix.mT.numpy(),  # precision.T = precision and is contiguous
                    self.weight.numpy(),
                    self.std.numpy().max(axis=-1) * k,
                    scale=scale,
                )
            )

            return torch.nn.functional.interpolate(
                truth.reshape((1, 1, self.size[0] // scale, self.size[1] // scale)), size=self.size, mode="bilinear"
            )[0, 0]

        # 3D
        truth = torch.tensor(
            numba_gmm_pdf_3d(
                self.size,
                self.mu.numpy(),
                self._distribution.precision_matrix.mT.numpy(),  # precision.T = precision and is contiguous
                self.weight.numpy(),
                self.std.numpy().max(axis=-1) * k,
                scale=scale,
            )
        )

        return torch.nn.functional.interpolate(
            truth.reshape((1, 1, self.size[0] // scale, self.size[1] // scale, self.size[2] // scale)),
            size=self.size,
            mode="trilinear",
        )[0, 0]

    def get_tracks_segmentation(self, min_value=0.3) -> torch.Tensor:
        """Build a ground truth segmentation image given the particles

        Args:
            min_value (float): Minimum value of the normalized pdf (max value is 1.0)
                to belong to the segmentation mask.

        Returns:
            torch.Tensor: Segmentation with 0 for background and i for each particle in [1, n]
                Shape: (H, W), dtype: int32
        """
        if self.dim == 2:
            return torch.tensor(
                _fast_segmentation_2d(
                    self.size,
                    self.mu.numpy(),
                    self._distribution.precision_matrix.mT.numpy(),
                    self.std.numpy().max(axis=-1) * np.float32(np.sqrt(-2 * np.log(min_value * 0.9))),
                    min_value,
                )
            )
        return torch.tensor(
            _fast_segmentation_3d(
                self.size,
                self.mu.numpy(),
                self._distribution.precision_matrix.mT.numpy(),
                self.std.numpy().max(axis=-1) * np.float32(np.sqrt(-2 * np.log(min_value * 0.9))),
                min_value,
            )
        )


class Recorder:
    """Record the state of particles in a running simulator

    Attributes:
        particles (GaussianParticles): Particles to record
        mu (torch.Tensor): Position of particles at each recorded time.
            Shape: (T, N, 2)
        theta (torch.Tensor): Angle of particles at each recorded time.
            Shape: (T, N)
        std (torch.Tensor): Size (std) of particles at each recorded time.
            Shape: (T, N, 2)
        weight (torch.Tensor): Weight of particles at each recorded time.
            Shape: (T, N)

    """

    def __init__(self, particles: GaussianParticles, size=50):
        self.particles = particles
        self._i = 0
        self._mu = torch.zeros((size, *particles.mu.shape), dtype=particles.mu.dtype)
        self._theta = torch.zeros((size, *particles.theta.shape), dtype=particles.theta.dtype)
        self._std = torch.zeros((size, *particles.std.shape), dtype=particles.std.dtype)
        self._weight = torch.zeros((size, *particles.weight.shape), dtype=particles.weight.dtype)

        self.update()

    @property
    def mu(self) -> torch.Tensor:
        return self._mu[: self._i]

    @property
    def theta(self) -> torch.Tensor:
        return self._theta[: self._i]

    @property
    def std(self) -> torch.Tensor:
        return self._std[: self._i]

    @property
    def weight(self) -> torch.Tensor:
        return self._weight[: self._i]

    def update(self):
        """Save a new state in the recorder"""
        if self._i >= len(self._mu):  # Reallocate twice the memory (dynamic arrays)
            self._mu = torch.cat((self._mu, torch.zeros_like(self._mu)))
            self._theta = torch.cat((self._theta, torch.zeros_like(self._theta)))
            self._std = torch.cat((self._std, torch.zeros_like(self._std)))
            self._weight = torch.cat((self._weight, torch.zeros_like(self._weight)))

        self._mu[self._i] = self.particles.mu
        self._theta[self._i] = self.particles.theta
        self._std[self._i] = self.particles.std
        self._weight[self._i] = self.particles.weight
        self._i += 1

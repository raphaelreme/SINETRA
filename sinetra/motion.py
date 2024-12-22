import dataclasses
from typing import List, Iterable, Optional

import numpy as np
import torch

import byotrack
import torch_tps

from . import optical_flow
from . import random
from . import springs
from .particle import GaussianParticles, _build_rot_2d, _build_rot_3d


@dataclasses.dataclass
class ShapeVariationConfig:
    """Configuration for ShapeVariation

    Attributes:
        period (float): Critical time of the variations (in number of frames)
        noise (float): Noise generated. (Multiplicative spring, see `ShapeVariation`)
    """

    period: float = 50.0
    noise: float = 0.0

    def build(self, **kwargs) -> "ShapeVariation":
        return ShapeVariation(kwargs["particles"], self.period, self.noise)


@dataclasses.dataclass
class LocalRotationConfig:
    """Configuration for LocalRotation

    Attributes:
        period (float): Critical time of the variations (in number of frames)
        noise (float): Noise generated
    """

    period: float = 50.0
    noise: float = 0.0

    def build(self, **kwargs) -> "LocalRotation":
        return LocalRotation(kwargs["particles"], self.period, self.noise)


@dataclasses.dataclass
class BrownianRotationConfig:
    """Configuration for BrownianRotation

    Attributes:
        noise (float): Additive gaussian noise to the angle (radian)
    """

    noise: float = 0.0

    def build(self, **_) -> "BrownianRotation":
        return BrownianRotation(self.noise)


@dataclasses.dataclass
class FlowMotionConfig:
    """Configuration for FlowMotion"""

    algorithm: str = "farneback"
    downscale: int = 4
    farneback: optical_flow.FarnebackConfig = dataclasses.field(default_factory=optical_flow.FarnebackConfig)

    def build(self, **kwargs) -> "FlowMotion":
        assert kwargs.get("video"), "Unable to create a flow motion without a true video input"

        if self.algorithm.lower() == "farneback":
            return FlowMotion(self.farneback.build(self.downscale), kwargs["video"], kwargs.get("start_frame", 0))

        raise ValueError(f"Unknown optical flow algorithm: {self.algorithm}.")


@dataclasses.dataclass
class RandomContractionConfig:
    """Random contraction forces configuration for the elastic motion

    Args:
        motion_rate (float): Expected number of contraction/elongation in a period T (critical time of the model)
        motion_size (int): Number of involved particles in each contraction is sampled from 2 to motion_size
        amplitude (float): Max magnitude of the displacement at each  (sampled from [amplitude / 2, amplitude])
        noise (float): Additional white noise handled by RandomNoise
    """

    motion_rate: float = 1.0
    motion_size: int = 10
    amplitude: float = 2.0
    noise: float = 0.0


@dataclasses.dataclass
class ElasticNoiseConfig:
    """Elastic noise configuration

    We currently only support the Random contraction forces
    """

    name: str = "contraction"
    contraction: RandomContractionConfig = dataclasses.field(default_factory=RandomContractionConfig)

    def build(self):
        if self.name.lower() == "contraction":
            return springs.RandomContraction(
                self.contraction.motion_rate,
                self.contraction.motion_size,
                self.contraction.amplitude,
                self.contraction.noise,
            )
        raise ValueError(f"Unknown Elastic force noise: {self.name}")


@dataclasses.dataclass
class ElasticMotionConfig:
    """Elastic motion configuration

    Attributes:
        alpha (float): Thin plate spline regularisation
        period (float): Critical time of the springs generated
        grid_step (int): Size of the grid for mass points.
        noise (ElasticNoiseConfig): The random force configuration

    """

    alpha: float = 10.0
    period: float = 50.0
    grid_step: int = 100
    noise: ElasticNoiseConfig = dataclasses.field(default_factory=ElasticNoiseConfig)

    def build(self, **kwargs) -> "ElasticMotion":
        points, neighbors = springs.RandomRelationalSprings.grid_springs_from_mask(kwargs["mask"], self.grid_step)
        quality = torch.tensor([0.5])
        w0 = torch.tensor([2 * torch.pi / self.period])
        random_accelerator = self.noise.build()
        spring = springs.RandomRelationalSprings(
            points, neighbors, w0**2, w0 / quality, random_accelerator=random_accelerator
        )
        return ElasticMotion(spring, self.alpha)


@dataclasses.dataclass
class MotionConfig:
    """Configuration of particle motions

    It defines a list of motions to apply to the simulations and the parameters
    for all motions (undefined motion are set to default values)
    """

    motions: List[str]
    shape_variation: ShapeVariationConfig = dataclasses.field(default_factory=ShapeVariationConfig)
    local_rotation: LocalRotationConfig = dataclasses.field(default_factory=LocalRotationConfig)
    brownian_rotation: BrownianRotationConfig = dataclasses.field(default_factory=BrownianRotationConfig)
    flow_motion: FlowMotionConfig = dataclasses.field(default_factory=FlowMotionConfig)
    elastic_motion: ElasticMotionConfig = dataclasses.field(default_factory=ElasticMotionConfig)

    def build(self, **kwargs) -> "MultipleMotion":
        motions = []
        for motion_name in self.motions:
            motion_name = motion_name.lower().strip()
            sub_cfg = getattr(self, motion_name)
            if motion_name in ("shape_variation", "local_rotation"):
                motions.append(sub_cfg.build(particles=kwargs["particles"]))
                if kwargs.get("background"):
                    motions.append(sub_cfg.build(particles=kwargs["background"]))

            motions.append(sub_cfg.build(**kwargs))

        return MultipleMotion(motions)


class BaseMotion:
    """Base motion Class

    A motion can be updated then applied (to background/particles)
    """

    def update(self) -> None:
        """Notify the motion that we have move to next frame"""

    def apply(self, particles: GaussianParticles) -> None:
        """Apply the motion to the particles

        It can moves mu, std or theta
        """

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: Optional[GaussianParticles]) -> None:
        for _ in range(warm_up):
            self.update()
            self.apply(particles)
            if background:
                self.apply(background)


class MultipleMotion(BaseMotion):
    """Handle multiple motions to apply to particles/backgrounds

    The current implementation of motions is not very robust. To prevent bugs, you should avoid
    having multiple motions handling the same parameter (mu, theta, std).
    """

    def __init__(self, motions: Iterable[BaseMotion]) -> None:
        super().__init__()
        self.motions = motions

    def update(self) -> None:
        for motion in self.motions:
            motion.update()

    def apply(self, particles: GaussianParticles) -> None:
        for motion in self.motions:
            motion.apply(particles)

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: Optional[GaussianParticles]) -> None:
        for motion in self.motions:
            motion.warm_up(warm_up, particles, background)


## Shape


class ShapeVariation(BaseMotion):
    """Create variation in shape (std)

    Shape of particles changes following  std_k = s_k * std_0
    Where s_k is a spring of equilibrium 1.0

    Will only modify the std of its particles
    """

    def __init__(self, particles: GaussianParticles, period=50.0, noise=0.0) -> None:
        super().__init__()
        self.particles_id = id(particles)
        self.std_0 = particles.std.clone()
        self.spring = springs.RandomAcceleratedSpring.build(
            torch.ones_like(particles.std),
            torch.tensor([0.5]),
            torch.tensor([noise]),
            torch.tensor([2 * torch.pi / period]),
        )

    def update(self) -> None:
        self.spring.update()

    def apply(self, particles: GaussianParticles) -> None:
        if id(particles) != self.particles_id:
            return

        particles.std = self.std_0 * self.spring.value


## Rotation


class BrownianRotation(BaseMotion):
    """Random rotation of particles

    The particles rotation follow a brownian motion (uncorrelated with the other rotations)
    """

    def __init__(self, noise=0.0) -> None:
        self.noise = noise

    def apply(self, particles: GaussianParticles) -> None:
        particles.theta += torch.randn(particles.theta.shape, generator=random.particle_generator) * self.noise


class LocalRotation(BaseMotion):
    """Local rotation of particles

    Each particle can rotate locally but not going to much further from the equilibrium point
    More rotation can be made by global rotation. (Thus rotation diff between particles is mostly kept)

    Will only modify the rotation of its particles
    """

    def __init__(self, particles: GaussianParticles, period=50.0, noise=0.0) -> None:
        self.particles_id = id(particles)
        self.spring = springs.RandomAcceleratedSpring.build(
            particles.theta,
            torch.tensor([0.5]),
            torch.tensor([noise]),
            torch.tensor([2 * torch.pi / period]),
        )

    def update(self) -> None:
        self.spring.update()

    def apply(self, particles: GaussianParticles) -> None:
        if id(particles) != self.particles_id:
            return

        particles.theta = self.spring.value


## Position


class BrownianPosition(BaseMotion):
    """Brownian motion of each particle"""

    def __init__(self, noise=0.0) -> None:
        self.noise = noise

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu += torch.randn(particles.mu.shape, generator=random.particle_generator) * self.noise


class FlowMotion(BaseMotion):
    """Motion based on real optical flow of a real animal"""

    def __init__(self, optflow: byotrack.OpticalFlow, video: byotrack.Video, start_frame=0) -> None:
        self.optflow = optflow
        self.video = video
        self.dir = 1
        self.frame_id = start_frame
        self.source = self.optflow.preprocess(self.video[self.frame_id])
        self.flow = np.zeros((2, 1, 1))

    def update(self) -> None:
        if not 0 <= self.frame_id + self.dir < len(self.video):
            self.dir = -self.dir  # If no more frame, let's go backward in the video

        self.frame_id += self.dir
        destination = self.optflow.preprocess(self.video[self.frame_id])

        self.flow = self.optflow.compute(self.source, destination)
        self.source = destination

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = torch.tensor(self.optflow.transform(self.flow, particles.mu.numpy())).to(torch.float32)

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: Optional[GaussianParticles]) -> None:
        pass  # No warmup with optical flow


class ElasticMotion(BaseMotion):
    """Elastic motion induced by RandomRelationalSprings

    Motion of particles are computed as a TPS interpolation of the springs points.
    """

    def __init__(self, spring: springs.RandomRelationalSprings, alpha=10.0) -> None:
        self.spring = spring
        self.tps = torch_tps.ThinPlateSpline(alpha)
        self.tps.fit(self.spring.points - self.spring.speeds, self.spring.points)

    def update(self) -> None:
        self.spring.update()
        self.tps.fit(self.spring.points - self.spring.speeds, self.spring.points)

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = self.tps.transform(particles.mu)


@dataclasses.dataclass
class GlobalMotionConfig:
    """Configuration for the global affine motions"""

    period: float = 1000.0
    noise_position: float = 0.0
    noise_theta: float = 0.0

    def build(self, particles: GaussianParticles) -> "GlobalDriftAndRotation":
        return GlobalDriftAndRotation(particles.mu.mean(dim=0), self.period, self.noise_position, self.noise_theta)


class GlobalDriftAndRotation:
    """Global drift and rotation for all particles (Rigid motion)

    Note: As it also plays with particles positions and angles at the same time as LocalRotation
          or ElasticMotion, it is designed to be reversible and applied just for imaging.

    Drift follow a spring (prevent all the particles to go out of focus) with a large period (slow vs the local motion)
    Rotation is also a spring (more for continuous reason) with the same period

    The operation is an affine transformation, in homegenous coordinates it can be written as:
    A = |R T|, where R is the rotation matrix (2D or 3D) and T the translation.
        |0 1|

    Here, we first do a rotation around the center of mass, then a translation to apply the transformation.
    """

    def __init__(self, mass_center: torch.Tensor, period=1000.0, noise_position=0.0, noise_theta=0.0) -> None:
        super().__init__()
        self.dim = mass_center.shape[0]
        self.translation_spring = springs.RandomAcceleratedSpring.build(
            torch.zeros_like(mass_center),
            torch.tensor([0.5]),
            torch.ones_like(mass_center) * noise_position,
            torch.tensor(2 * torch.pi / period),
        )
        self.rotation_spring = springs.RandomAcceleratedSpring.build(
            torch.zeros(1 if self.dim == 2 else 3),
            torch.tensor([0.5]),
            torch.full((1 if self.dim == 2 else 3,), noise_theta),
            torch.tensor(2 * torch.pi / period),
        )
        self.mass_center = mass_center

    @property
    def translation(self) -> torch.Tensor:
        return self.translation_spring.value

    @property
    def theta(self) -> torch.Tensor:
        return self.rotation_spring.value

    @property
    def affine(self) -> torch.Tensor:
        if self.dim == 2:
            rotation = _build_rot_2d(self.theta[None])[0]
        else:
            rotation = _build_rot_3d(self.theta[None])[0]

        # The true translation is the translation induced by the rotation plus the actual translation
        translation = self.translation + self.mass_center - rotation @ self.mass_center

        affine = torch.eye(self.dim + 1)
        affine[: self.dim, : self.dim] = rotation
        affine[: self.dim, self.dim] = translation
        return affine

    def update(self) -> None:
        self.translation_spring.update()
        self.rotation_spring.update()

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = self.apply_tensor(particles.mu)
        particles.theta += self.theta

    def apply_tensor(self, points: torch.Tensor) -> torch.Tensor:
        homogeneous = torch.ones((points.shape[0], self.dim + 1))
        homogeneous[:, : self.dim] = points
        homogeneous = homogeneous @ self.affine.T

        return homogeneous[:, : self.dim]

    def revert(self, particles: GaussianParticles) -> None:
        particles.mu = self.revert_tensor(particles.mu)
        particles.theta -= self.theta

    def revert_tensor(self, points: torch.Tensor) -> torch.Tensor:
        homogeneous = torch.ones((points.shape[0], self.dim + 1))
        homogeneous[:, : self.dim] = points
        homogeneous = homogeneous @ torch.inverse(self.affine).T

        return homogeneous[:, : self.dim]

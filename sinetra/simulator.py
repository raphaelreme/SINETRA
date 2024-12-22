import dataclasses
from typing import Optional, Tuple

import torch

import byotrack

from . import random
from .emission import EmissionModel, EmissionConfig
from .mask import random_mask, mask_from_frame
from .motion import GlobalDriftAndRotation, GlobalMotionConfig, MotionConfig, MultipleMotion
from .particle import GaussianParticles, GaussianParticlesConfig


@dataclasses.dataclass
class ImagingConfig:
    """Imaging configuration

    Attributes:
        delta (float): Integration time (Controls the poisson shot noise on the image)
        alpha (float): Ponderation of particles over background (Linearly mixed: I = aP + (1-a)B)
        noise (float): Noise baseline to add to the background (even outside of the background)
    """

    # psnr (float): Peak Signal to Noise Ratio (Controls the intensity of the particles over the background)

    delta: float = 100.0
    alpha: float = 0.2
    # psnr: float = 1.5
    noise: float = 0.1


@dataclasses.dataclass
class VideoConfig:
    """Video configuration

    Attributes:
        path (str): Path to a valid video readable by ByoTrack
        start (int): Starting frame
        stop (int): Last frame
        step (int): go through the video `step` frame at a time.
        randomise (bool): Start at a random frame (inside start:stop) and use a random
            step in (-1, 1)

    """

    path: str = ""
    start: int = 0
    stop: int = -1
    step: int = 1
    randomise: bool = False

    transform: byotrack.VideoTransformConfig = dataclasses.field(default_factory=byotrack.VideoTransformConfig)

    def open(self) -> Optional[byotrack.Video]:
        if self.path == "":
            return None

        video = byotrack.Video(self.path)[slice(self.start, self.stop, self.step)]
        video.set_transform(self.transform)
        return video


@dataclasses.dataclass
class SimulatorConfig:
    """Simulator configuration

    Configure the whole simulator with imaging, particles, background, motion and emission models.
    """

    shape: Tuple[int, ...]
    base_video: VideoConfig
    imaging_config: ImagingConfig
    particle: GaussianParticlesConfig
    background: GaussianParticlesConfig
    emission: EmissionConfig
    motion: MotionConfig
    global_motion: GlobalMotionConfig
    warm_up: int = 500


class Simulator:
    """Simulator object

    Handle the image generation and temporal evolution.
    """

    def __init__(
        self,
        particles: GaussianParticles,
        motion: MultipleMotion,
        imaging_config: ImagingConfig,
        *,
        background: Optional[GaussianParticles] = None,
        emission_model: Optional[EmissionModel] = None,
        global_motion: Optional[GlobalDriftAndRotation] = None,
        background_gain: Optional[float] = None,
    ):
        self.particles = particles
        self.background = background
        if background_gain is None:
            self.background_gain = background.draw_truth(scale=4).max().item() if background else 1.0
        else:
            self.background_gain = background_gain

        self.emission_model = emission_model
        self.motion = motion
        self.global_motion = global_motion
        self.imaging_config = imaging_config

    def generate_image(self):
        particles = self.particles.draw_truth()

        if self.background:
            background = self.background.draw_truth(scale=4)
            background /= self.background_gain
            background.clip_(0.0, 1.0)
            background = (
                1 - self.imaging_config.noise
            ) * background + self.imaging_config.noise  # Add a noise baseline
        else:
            background = self.imaging_config.noise * torch.ones_like(particles)

        # snr = 10 ** (self.imaging_config.psnr / 10)
        # alpha = (snr - 1) / (
        #     snr - 1 + 1 / 0.6
        # )  # Uses E[B(z_p)] = 0.6 and assume that the Poisson Shot noise is negligeable in the SNR
        baseline = (1 - self.imaging_config.alpha) * background + self.imaging_config.alpha * particles

        # Poisson shot noise
        image = (
            torch.poisson(self.imaging_config.delta * baseline, generator=random.imaging_generator)
            / self.imaging_config.delta
        )

        image.clip_(0.0, 1.0)

        return image

    def update(self):
        if self.emission_model is not None:
            self.emission_model.update()

        self.motion.update()

        if self.global_motion:
            self.global_motion.revert(self.particles)
            if self.background:
                self.global_motion.revert(self.background)

        self.motion.apply(self.particles)
        if self.background:
            self.motion.apply(self.background)

        if self.global_motion:
            self.global_motion.update()
            self.global_motion.apply(self.particles)
            if self.background:
                self.global_motion.apply(self.background)

        self.particles.build_distribution()
        if self.background:
            self.background.build_distribution()

    @staticmethod
    def from_config(config: SimulatorConfig) -> "Simulator":  # pylint: disable=too-many-branches
        video = config.base_video.open()

        start_frame = 0
        if video is None:
            mask = random_mask(config.shape)
        else:
            t_step = 1
            # Randomise the way the video is read
            if config.base_video.randomise:
                start_frame = int(torch.randint(0, len(video), (1,), generator=random.particle_generator).item())

                t_step = 1 if torch.rand(1, generator=random.particle_generator) > 0.5 else -1
                i_step = 1 if torch.rand(1, generator=random.particle_generator) > 0.5 else -1
                j_step = 1 if torch.rand(1, generator=random.particle_generator) > 0.5 else -1
                video = video[::t_step, ::i_step, ::j_step]

            frames = [config.base_video.start, config.base_video.stop]
            if t_step == 1:
                print(f"Reading video from the {start_frame}-th frame (inside {frames}) in a direct order")
            else:
                print(
                    f"Reading video from the {len(video) - start_frame}-th frame (inside {frames}) in a reversed order"
                )

            mask = mask_from_frame(video[start_frame])

        particles = GaussianParticles(config.particle.n, mask, config.particle.min_std, config.particle.max_std)
        if config.particle.min_dist > 0:
            particles.filter_close_particles(config.particle.min_dist)

        background = None
        if config.background.n > 0:
            background = GaussianParticles(
                config.background.n, mask, config.background.min_std, config.background.max_std
            )
            if config.background.min_dist > 0:
                background.filter_close_particles(config.background.min_dist)

        emission_model = config.emission.build(particles)

        motion = config.motion.build(
            video=video, mask=mask, particles=particles, background=background, start_frame=start_frame
        )

        global_motion = None
        if config.global_motion.noise_position > 0 or config.global_motion.noise_theta > 0:
            global_motion = GlobalDriftAndRotation(
                particles.mu.mean(dim=0),
                config.global_motion.period,
                config.global_motion.noise_position,
                config.global_motion.noise_theta,
            )

        # Warmup and find gain as the max of 5 frames during the warmup
        if background:
            gain = background.draw_truth(scale=4).max().item()
            for _ in range(5):
                motion.warm_up(config.warm_up // 5, particles, background)  # Update and apply motion
                background.build_distribution()
                gain = max(background.draw_truth(scale=4).max().item(), gain)
        else:
            motion.warm_up(config.warm_up, particles, background)

        for _ in range(config.warm_up):
            emission_model.update()  # Update nam

            if global_motion:  # Update global motional motion as it is first reverted on simulator.update
                global_motion.update()

        if global_motion:  # UGLY: Apply glob
            global_motion.apply(particles)
            if background:
                global_motion.apply(background)

        # Update distributions
        particles.build_distribution()
        if background:
            background.build_distribution()

        simulator = Simulator(
            particles,
            motion,
            config.imaging_config,
            background=background,
            emission_model=emission_model,
            global_motion=global_motion,
            background_gain=gain,
        )
        # True warm up, but expensive with Optical flow
        # for _ in tqdm.trange(config.warm_up, desc="Warming up"):
        #     simulator.update()

        return simulator

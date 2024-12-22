from __future__ import annotations

import dataclasses
import torch

from . import particle
from . import random


@dataclasses.dataclass
class NamConfig:
    """Neuronal activity configuration"""

    firing_rate: float = 0.01
    decay: float = 0.95


@dataclasses.dataclass
class EmissionConfig:
    """Parameters for EmissionModel"""

    mode: str = "constant"
    nam: NamConfig = dataclasses.field(default_factory=NamConfig)

    def build(self, particles: particle.GaussianParticles) -> EmissionModel:
        if self.mode.lower() == "nam":
            return NeuronalActivityModel(particles, self.nam.firing_rate, self.nam.decay)

        return EmissionModel(particles)


class EmissionModel:
    """Default class for particles emissions models (that control how luminous particles are)

    Keep a constant intensity weight by default.

    Attributes:
        particles (GaussianParticles): The particles to handle
        generator (torch.Generator): Random generator used for randomness

    """

    def __init__(self, particles: particle.GaussianParticles):
        self.particles = particles
        self.generator = random.emission_generator

    def update(self, dt=1.0) -> None:
        """Update the weights of the particles after dt"""


class NeuronalActivityModel(EmissionModel):  # pylint: disable=too-few-public-methods
    """Simple Neuronal Activity Model (nam)

    The neurons are supposed independent following a simple generation process:
    i(t+1) = decay * i(t) + gain * firing

    We add some complexity to prevent some behaviors:
    - The gain of each firing depends on the actual value of the neurons
        (Large value, small gain to prevent reaching MAX_WEIGHT)
    - The added value is then sampled from N(gain, (gain / 5)**2)
    - The real weights retained for neurons intensity is an EMA of the computed weights
        This is done to mimic natural firings where the intensity does not jump in a single frame

    Attributes:
        particles (GaussianParticles): The particles to handle
        firing_rate (float): Firing rates of particles
        decay (float): Exponential decay of the weights
        immediate_weights (torch.Tensor): Weights without the EMA (Smoothing)

    """

    # pylint: disable=invalid-name
    MAX_WEIGHT = 2
    MIN_WEIGHT = 0.1  # Minimal baseline for particles
    FIRING_GAIN = 1.0
    SMOOTHING_DECAY = 0.75  # EMA factor to prevent hard firings
    # pylint: enable=invalid-name

    def __init__(self, particles: particle.GaussianParticles, firing_rate=0.01, decay=0.95):
        super().__init__(particles)
        self.firing_rate = firing_rate
        self.decay = decay
        self.immediate_weight = particles.weight.clone()

    def update(self, dt=1.0):
        self.immediate_weight *= self.decay * dt  # Decay the weight
        firing = torch.rand(self.immediate_weight.shape, generator=self.generator) < self.firing_rate * dt
        gain = self.FIRING_GAIN - self.immediate_weight[firing] * (self.FIRING_GAIN / self.MAX_WEIGHT)
        self.immediate_weight[firing] += gain + torch.randn(firing.sum(), generator=self.generator) * gain * 0.2

        # Update the particles weights as an EMA with a gamma of SMOOTHING_DELAY
        self.particles.weight.sub_(
            (1.0 - self.SMOOTHING_DECAY * dt) * (self.particles.weight - self.immediate_weight - self.MIN_WEIGHT)
        )

        # NOTE: We could add some gaussian noise to weights

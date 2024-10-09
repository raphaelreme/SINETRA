import dataclasses
import enum


import cv2
import torch
import tqdm

import byotrack
from byotrack.implementation.detector.wavelet import WaveletDetector


class DetectionMethod(enum.Enum):
    """Implemented detections method"""

    WAVELET = "wavelet"
    FAKE = "fake"


@dataclasses.dataclass
class WaveletConfig:
    """Wavelet detection configuration"""

    k: float = 2.5
    scale: int = 1
    min_area: float = 5.0


@dataclasses.dataclass
class FakeConfig:
    fpr: float = 0.2  # Bad detection rate
    fnr: float = 0.2  # Miss detection rate
    measurement_noise: float = 0.5


@dataclasses.dataclass
class DetectionConfig:
    """Detection configuration"""

    detector: DetectionMethod
    wavelet: WaveletConfig
    fake: FakeConfig

    def create_detector(self, mu: torch.Tensor) -> byotrack.Detector:
        if self.detector == DetectionMethod.WAVELET:
            return WaveletDetector(self.wavelet.scale, self.wavelet.k, self.wavelet.min_area)

        return FakeDetector(mu, self.fake.measurement_noise, self.fake.fpr, self.fake.fnr, False)


class FakeDetector(byotrack.Detector):  # TODO: include weight, std and theta to detect more closely what is expected
    """Fake detections from ground truth

    The current implementation is a bit simple and generate only squared roi around a random position
    """

    def __init__(self, mu: torch.Tensor, noise=0.5, fpr=0.2, fnr=0.2, generate_outside_particles=False):
        self.size = 5
        self.noise = noise
        self.fpr = fpr
        self.fnr = fnr
        self.mu = mu
        self.n_particles = mu.shape[1]
        self.generate_outside_particles = generate_outside_particles

    def run(self, video):
        detections_sequence = []

        for k, frame in enumerate(tqdm.tqdm(video)):
            frame = frame[..., 0]  # Drop channel
            shape = torch.tensor(frame.shape)

            detected = torch.rand(self.n_particles) >= self.fnr  # Miss some particles (randomly)

            idx = torch.arange(self.n_particles)[detected]
            positions = self.mu[k, detected] + torch.randn((detected.sum(), 2)) * self.noise

            valid = torch.logical_and((positions > 0).all(dim=-1), (positions < shape - 1).all(dim=-1))
            positions = positions[valid]
            idx = idx[valid]

            # Create fake detections
            # 1- Quickly compute the background mask (could be smarter or based on the true initial mask ?)
            mask = torch.tensor(cv2.GaussianBlur(frame, (33, 33), 15) > 0.2)
            mask_proportion = mask.sum().item() / mask.numel()

            # 2- Scale fpr by the mask proportion
            n_fake = int(len(positions) * (self.fpr + torch.randn(1).item() * self.fpr / 10) / mask_proportion)
            false_alarm = torch.rand(n_fake, 2) * (shape - 1)

            if not self.generate_outside_particles:  # Filter fake detections outside the mask
                false_alarm = false_alarm[mask[false_alarm.long()[:, 0], false_alarm.long()[:, 1]]]

            positions = torch.cat((positions, false_alarm))

            # Ground truth idx added from debug if required
            idx = torch.cat((idx, -torch.ones_like(false_alarm)[:, 0]))

            size = torch.where(torch.round(positions * 2) % 2 == 0, self.size, self.size - 1)
            bbox = torch.cat((torch.ceil(torch.round(positions * 2) / 2) - size // 2, size), dim=-1).to(torch.int32)

            detections_sequence.append(
                byotrack.Detections(
                    {
                        # "position": positions,  # KOFT is able to use the subpixelic info, not the others
                        "bbox": bbox.round().to(torch.int32),
                        "shape": shape,
                        "idx": idx,
                    },
                    frame_id=k,
                )
            )

        return detections_sequence

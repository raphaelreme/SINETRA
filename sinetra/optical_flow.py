"""Optical flow to use in the simulation"""

import dataclasses

import cv2

import byotrack
import byotrack.implementation.optical_flow.opencv


@dataclasses.dataclass
class FarnebackConfig:
    """Configuration for Farneback algorithm"""

    win_size: int = 20

    def build(self, downscale: int) -> byotrack.OpticalFlow:
        return byotrack.implementation.optical_flow.opencv.OpenCVOpticalFlow(
            cv2.FarnebackOpticalFlow.create(winSize=self.win_size), downscale=downscale
        )


@dataclasses.dataclass
class OpticalFlowConfig:
    """Optical flow configuration"""

    name: str = "farneback"
    downscale: int = 4
    farneback: FarnebackConfig = dataclasses.field(default_factory=FarnebackConfig)

    def build(self) -> byotrack.OpticalFlow:
        if self.name.lower() == "farneback":
            return self.farneback.build(self.downscale)

        raise ValueError(f"Unknown optical flow: {self.name}.")

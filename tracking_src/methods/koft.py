"""Code and config to run koft"""

import dataclasses

import cv2
import skimage

import byotrack
import byotrack.implementation.optical_flow.opencv
import byotrack.implementation.optical_flow.skimage
from byotrack.implementation.linker.frame_by_frame import koft


@dataclasses.dataclass
class FarnebackConfig:
    """Configuration for Farneback algorithm (OpenCV, only 2D)"""

    win_size: int = 20

    def build(self, downscale: int) -> byotrack.OpticalFlow:
        return byotrack.implementation.optical_flow.opencv.OpenCVOpticalFlow(
            cv2.FarnebackOpticalFlow.create(winSize=self.win_size), downscale=downscale
        )


@dataclasses.dataclass
class TVL1Config:
    """Configuration for TVL1 algorithm (skimage, supports 3D)"""

    attachment: float = 5.0
    num_warp: int = 2

    def build(self, downscale: int) -> byotrack.OpticalFlow:
        return byotrack.implementation.optical_flow.skimage.SkimageOpticalFlow(
            skimage.registration.optical_flow_tvl1,
            downscale=downscale,
            parameters={"attachment": self.attachment, "num_warp": self.num_warp},
        )


@dataclasses.dataclass
class OpticalFlowConfig:
    """Optical flow configuration"""

    name: str = "farneback"
    downscale: int = 4
    farneback: FarnebackConfig = FarnebackConfig()
    tvl1: TVL1Config = TVL1Config()

    def build(self) -> byotrack.OpticalFlow:
        if self.name.lower() == "farneback":
            return self.farneback.build(self.downscale)
        if self.name.lower() == "tvl1":
            return self.tvl1.build(self.downscale)

        raise ValueError(f"Unknown optical flow: {self.name}.")


@dataclasses.dataclass
class KOFTConfig(koft.KOFTLinkerParameters):
    """Configuration for KOFT algorithm"""

    optical_flow: OpticalFlowConfig = OpticalFlowConfig()

    def build(self) -> koft.KOFTLinker:
        return koft.KOFTLinker(self, self.optical_flow.build())

import dataclasses
import pathlib
import shutil
import warnings

import cv2
import dacite
import numpy as np
import tifffile  # type: ignore
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

from sinetra.motion import ElasticMotion
from sinetra.particle import Recorder
from sinetra.random import enforce_all_seeds
from sinetra.simulator import Simulator, SimulatorConfig


@dataclasses.dataclass
class MainConfig:
    """Experiment configuration"""

    seed: int
    n_frames: int
    display: bool
    simulator: SimulatorConfig
    dataset_path: pathlib.Path
    format: str = "full"  # Generates annotations as .pt and .tiff


def main(name: str, cfg_data: dict) -> None:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(MainConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple]))

    pathlib.Path("tracks").mkdir()

    dataset_path = cfg.dataset_path / name
    if dataset_path.exists():
        warnings.warn("This dataset was already generated, it will be removed and replaced")
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=False)

    enforce_all_seeds(cfg.seed)

    simulator = Simulator.from_config(cfg.simulator)
    recorder = Recorder(simulator.particles)

    # Find springs for display and save
    springs = None
    for motion in simulator.motion.motions:
        if isinstance(motion, ElasticMotion):
            springs = motion.spring

    # Old writer
    # Create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    # writer = cv2.VideoWriter("video.mp4", fourcc, 30, simulator.particles.size, isColor=False)

    # 2D/3D compatible write with tiff, but we need to allocate the full video
    video = np.zeros((cfg.n_frames, *cfg.simulator.shape), dtype=np.uint8)

    frame_saved = False
    k = 0
    try:
        for k in tqdm.trange(cfg.n_frames):
            frame_saved = False
            frame = simulator.generate_image().numpy()
            if cfg.format == "full":
                segmentation = simulator.particles.get_tracks_segmentation().numpy().astype(np.uint16)
                # Add axes to match imagej tiff default format: TZCYXS
                if segmentation.ndim == 2:
                    segmentation = segmentation[None, None, None, ..., None]
                else:
                    segmentation = segmentation[None, :, None, ..., None]

                tifffile.imwrite(f"tracks/{k:04}.tiff", segmentation, imagej=True)

            # writer.write((frame * 255).astype(np.uint8))
            video[k] = (frame * 255).round().astype(np.uint8)
            frame_saved = True

            if cfg.display and frame.ndim == 2:
                if springs:  # Display also the springs
                    points = springs.points
                    if simulator.global_motion:
                        points = simulator.global_motion.apply_tensor(springs.points)
                    for i, j in points.round().to(torch.int32).tolist():
                        cv2.circle(frame, (j, i), 2, 255, -1)  # type: ignore

                cv2.imshow("Frame", frame)
                cv2.setWindowTitle("Frame", f"Frame {k}/{cfg.n_frames}")
                cv2.waitKey(delay=1)

            simulator.update()
            recorder.update()

    finally:
        if video.ndim == 3:
            tifffile.imwrite("video.tiff", video[: k + frame_saved, None, None, ..., None], imagej=True)  # TZCYXS
        else:
            tifffile.imwrite("video.tiff", video[: k + frame_saved, :, None, ..., None], imagej=True)  # TZCYXS

        torch.save(
            {
                "mu": recorder.mu[: k + frame_saved],
                "theta": recorder.theta[: k + frame_saved],
                "std": recorder.std[: k + frame_saved],
                "weight": recorder.weight[: k + frame_saved],
            },
            "video_data.pt",
        )

        if cfg.format == "full":
            # Save the weights in a readable format for anyone
            # (The segmentation is already saved in tiff)
            np.savetxt("weights.txt", recorder.weight[: k + frame_saved].numpy(), fmt="%.4f", encoding="utf-8")

        cv2.destroyAllWindows()
        # writer.release()

        # Copy the useful data to the dataset folder
        shutil.copy("video.tiff", dataset_path / "video.tiff")
        shutil.copy("video_data.pt", dataset_path / "video_data.pt")

        if cfg.format == "full":
            shutil.copy("weights.txt", dataset_path / "weights.txt")
            shutil.copytree("tracks", dataset_path / "tracks")

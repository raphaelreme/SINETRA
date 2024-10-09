import dataclasses
import pathlib
import shutil

import cv2
import dacite
import numpy as np
import tifffile
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

from neuro_track.motion import ElasticMotion
from neuro_track.particle import Recorder
from neuro_track.random import enforce_all_seeds
from neuro_track.simulator import Simulator, SimulatorConfig


@dataclasses.dataclass
class MainConfig:
    """Experiment configuration"""

    seed: int
    n_frames: int
    display: bool
    simulator: SimulatorConfig
    dataset_path: pathlib.Path


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(MainConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple]))

    pathlib.Path("tracks").mkdir()

    dataset_path = cfg.dataset_path / name
    dataset_path.mkdir(parents=True, exist_ok=False)

    enforce_all_seeds(cfg.seed)

    simulator = Simulator.from_config(cfg.simulator)
    recorder = Recorder(simulator.particles)

    # Lets print the alpha used for the simulation (mixture coef between background and particles)
    snr = 10 ** (cfg.simulator.imaging_config.psnr / 10)
    alpha = (snr - 1) / (snr - 1 + 1 / 0.6)  # From simulator
    print("Alpha:", alpha)

    # Find springs for display and save
    springs = None
    for motion in simulator.motion.motions:
        if isinstance(motion, ElasticMotion):
            springs = motion.spring

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter("video.mp4", fourcc, 30, simulator.particles.size, isColor=False)

    frame_saved = False
    k = 0
    try:
        for k in tqdm.trange(cfg.n_frames):
            frame_saved = False
            frame = simulator.generate_image().numpy()
            segmentation = simulator.particles.get_tracks_segmentation().numpy().astype(np.uint16)
            # Add axes to match imagej tiff default format: TZCXY
            if segmentation.ndim == 2:
                segmentation = segmentation[None, None, None, ..., None]
            else:
                segmentation = segmentation[None, :, None, ..., None]

            writer.write((frame * 255).astype(np.uint8))
            tifffile.imwrite(f"tracks/{k:04}.tiff", segmentation, imagej=True)
            frame_saved = True

            if cfg.display:
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
        torch.save(
            {
                "mu": recorder.mu[: k + frame_saved],
                "theta": recorder.theta[: k + frame_saved],
                "std": recorder.std[: k + frame_saved],
                "weight": recorder.weight[: k + frame_saved],
            },
            "video_data.pt",
        )
        # Save the weights in a readable format for anyone
        # (The segmentation is already saved in tiff)
        np.savetxt("weights.txt", recorder.weight[: k + frame_saved].numpy(), fmt="%.4f", encoding="utf-8")

        cv2.destroyAllWindows()
        writer.release()

        # Copy the useful data to the dataset folder
        shutil.copy("video.mp4", dataset_path / "video.mp4")
        shutil.copy("video_data.pt", dataset_path / "video_data.pt")
        shutil.copy("weights.txt", dataset_path / "weights.txt")
        shutil.copytree("tracks", dataset_path / "tracks")

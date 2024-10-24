"""Small grid search to find an almost optimal wavelet detection method"""

import dataclasses
import enum
import pathlib
from typing import List

import dacite
import tqdm  # type: ignore
import yaml  # type: ignore

from . import data, detect
from .metrics import detection as detection_metrics
from .utils import enforce_all_seeds


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    data: pathlib.Path
    detection: detect.DetectionConfig
    scales: List[int]
    k: List[float]
    step: int = 4


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(
        ExperimentConfig,
        cfg_data,
        dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]),
    )

    enforce_all_seeds(cfg.seed)

    video = data.open_video(cfg.data)
    ground_truth = data.load_ground_truth(cfg.data)

    best_f1 = 0.0
    best_scale = 0
    best_k = 0.0
    # Detections
    for scale in tqdm.tqdm(cfg.scales):
        for k in tqdm.tqdm(cfg.k):
            detector = detect.WaveletDetector(scale, k, min_area=cfg.detection.wavelet.min_area, batch_size=5)

            detections_sequence = detector.run(video[:: cfg.step])  # Let's detect every four frames to speed up

            # Evaluate detections step performances
            tp = 0.0
            n_pred = 0.0
            n_true = 0.0
            for i, detections in enumerate(detections_sequence):
                det_metrics = detection_metrics.DetectionMetric(2.0).compute_at(
                    detections, ground_truth["mu"][:: cfg.step][i], ground_truth["weight"][:: cfg.step][i]
                )
                tp += det_metrics["tp"]
                n_pred += det_metrics["n_pred"]
                n_true += det_metrics["n_true"]

            recall = tp / n_true if n_true else 1.0
            precision = tp / n_pred if n_pred else 1.0
            f1 = 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0

            tqdm.tqdm.write(f"Scale: {scale} and K: {k} => R, P, F1: {recall, precision, f1}")

            if f1 > best_f1:
                best_f1 = f1
                best_scale = scale
                best_k = k

    print(f"Best: Scale={best_scale}, K={best_k} => F1={best_f1}")

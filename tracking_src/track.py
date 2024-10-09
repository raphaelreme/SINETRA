import dataclasses
import pathlib
from typing import List

import dacite
import enum
import tqdm  # type: ignore
import yaml  # type: ignore

import byotrack
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

from . import data, detect
from .methods.koft import KOFTConfig
from .methods.emht import EMHTConfig, icy_emht
from .methods.trackmate import TrackMateConfig
from .metrics import detection as detection_metrics, tracking as tracking_metrics
from .utils import enforce_all_seeds


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    data: pathlib.Path
    tracking_methods: List[str]
    detection: detect.DetectionConfig
    koft: KOFTConfig
    emht: EMHTConfig
    trackmate: TrackMateConfig


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(
        ExperimentConfig,
        cfg_data,
        dacite.Config(
            cast=[pathlib.Path, tuple, enum.Enum], type_hooks={icy_emht.Motion: lambda s: icy_emht.Motion[s.upper()]}
        ),
    )

    enforce_all_seeds(cfg.seed)

    print(cfg.koft)

    video = data.open_video(cfg.data)
    ground_truth = data.load_ground_truth(cfg.data)

    # Detections
    detector = cfg.detection.create_detector(ground_truth["mu"])
    detections_sequence = detector.run(video)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for detections in detections_sequence:
        det_metrics = detection_metrics.DetectionMetric(1.5).compute_at(
            detections, ground_truth["mu"][detections.frame_id], ground_truth["weight"][detections.frame_id]
        )
        tp += det_metrics["tp"]
        n_pred += det_metrics["n_pred"]
        n_true += det_metrics["n_true"]

    print("=======Detection======")
    print("Recall", tp / n_true if n_true else 1.0)
    print("Precision", tp / n_pred if n_pred else 1.0)
    print("f1", 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0)

    refiner = ForwardBackwardInterpolater()
    metrics = {}
    for method in tqdm.tqdm(cfg.tracking_methods):
        linker: byotrack.Linker = getattr(cfg, method).build()
        try:
            tracks = linker.run(video, detections_sequence)
            tracks = refiner.run(video, tracks)  # Close gap (for u-track, EMHT)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            tqdm.tqdm.write(str(exc))
            tracks = []  # Tracking failed (For instance: timeout in EMHT)

        tqdm.tqdm.write(f"Built {len(tracks)} tracks")

        if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 20:
            tqdm.tqdm.write(f"{method} failed (too few or too many tracks). Continuing...")
            continue

        hota = tracking_metrics.compute_tracking_metrics(tracks, ground_truth)

        # Hota @ 2 (-8 => Thresholds is 2)
        metrics[method] = {key: value[-8].item() for key, value in hota.items()}
        byotrack.Track.save(tracks, f"{method}_tracks.pt")

        tqdm.tqdm.write(f"{method} => HOTA@2.0: {metrics[method]['HOTA']}")

    with open("metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics))

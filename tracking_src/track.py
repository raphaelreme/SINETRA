import copy
import dataclasses
import enum
import pathlib
import time
from typing import List

import dacite
import tqdm  # type: ignore
import yaml  # type: ignore

import byotrack
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

from . import data, detect
from .methods.koft import KOFTConfig
from .methods.emht import EMHTConfig, icy_emht
from .methods.trackmate import TrackMateConfig
from .methods.zephir import ZephIRConfig
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
    zephir: ZephIRConfig


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

    video = data.open_video(cfg.data)
    ground_truth = data.load_ground_truth(cfg.data)

    # Detections
    detector = cfg.detection.create_detector(ground_truth["mu"])
    detections_sequence = detector.run(video)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for i, detections in enumerate(detections_sequence):
        det_metrics = detection_metrics.DetectionMetric(2.0).compute_at(
            detections, ground_truth["mu"][i], ground_truth["weight"][i]
        )
        tp += det_metrics["tp"]
        n_pred += det_metrics["n_pred"]
        n_true += det_metrics["n_true"]

    print("=======Detection======")
    print("Recall", tp / n_true if n_true else 1.0)
    print("Precision", tp / n_pred if n_pred else 1.0)
    print("f1", 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0)

    linker: byotrack.Linker
    refiner = ForwardBackwardInterpolater()
    metrics = {}
    for method in tqdm.tqdm(cfg.tracking_methods):
        if method == "zephir-low":  # Specific behavior for zephir-low
            cfg_ = copy.deepcopy(cfg.zephir)
            cfg_.num_annotated_frames = 3
            linker = cfg_.build()
        else:
            linker = getattr(cfg, method).build()

        t = time.time()
        try:
            tracks = linker.run(video, detections_sequence)
            tracks = refiner.run(video, tracks)  # Close gap (for u-track, EMHT)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            tqdm.tqdm.write(str(exc))
            tracks = []  # Tracking failed (For instance: timeout in EMHT)

        t = time.time() - t

        tqdm.tqdm.write("\n")

        # FPS are not so much fair for eMHT and TrackMate.
        # In the paper, we manually measure the time from when the tracking in Java truly starts
        # and up to when it truly ends, removing the python wraping time.
        tqdm.tqdm.write(f"Built {len(tracks)} tracks in {t} seconds ({len(video) / t} fps)")

        if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 20:
            tqdm.tqdm.write(f"{method} failed (too few or too many tracks). Continuing...\n")
            continue

        hota = tracking_metrics.compute_tracking_metrics(tracks, ground_truth)

        # Hota @ 2 (-8 => Thresholds is 2)
        metrics[method] = {key: value[-8].item() for key, value in hota.items()}
        byotrack.Track.save(tracks, f"{method}_tracks.pt")

        tqdm.tqdm.write(f"{method} => HOTA@2.0: {metrics[method]['HOTA']}\n")

    with open("metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics))

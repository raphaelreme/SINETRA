"""Should not be used for tracking results, only for choosing the parameters

Note: we limit to 10 configurations to test because on real data, testing tracking parameters
is hard (either because the user does not know which parameters to choose, or because no ground
truth is available and only a visual check can be made, or because it takes too much time)
"""

import copy
import dataclasses
import enum
import pathlib
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
    tracking_method: str
    detection: detect.DetectionConfig
    koft: KOFTConfig
    emht: EMHTConfig
    trackmate: TrackMateConfig
    zephir: ZephIRConfig
    num_frames: int
    keys: List[str]
    values: List[List]


def modify_dataclass(dataclass, keys: List[str], values: List[List]):
    dataclass = copy.deepcopy(dataclass)
    for key, value in zip(keys, values):
        *attrs, final = key.split(".")
        dataclass_ = dataclass
        for attr in attrs:
            dataclass_ = getattr(dataclass_, attr)

        setattr(dataclass_, final, value)

    return dataclass


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

    video = data.open_video(cfg.data)[: cfg.num_frames]
    ground_truth = data.load_ground_truth(cfg.data)
    ground_truth["mu"] = ground_truth["mu"][: cfg.num_frames]
    ground_truth["weight"] = ground_truth["weight"][: cfg.num_frames]

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

    parameters = getattr(cfg, cfg.tracking_method)

    parameters_list = [parameters] + [modify_dataclass(parameters, cfg.keys, values) for values in cfg.values]
    if len(parameters_list) > 11:
        raise ValueError("You should not try to much parameters. In real life this cannot be done.")

    refiner = ForwardBackwardInterpolater()
    best_values = []
    best_parameters = parameters
    best_hota = 0.0
    for linker_parameters, values in zip(tqdm.tqdm(parameters_list), [[]] + cfg.values):
        linker: byotrack.Linker = linker_parameters.build()
        try:
            tracks = linker.run(video, detections_sequence)
            tracks = refiner.run(video, tracks)  # Close gap (for u-track, EMHT)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            tqdm.tqdm.write(str(exc))
            tracks = []  # Tracking failed (For instance: timeout in EMHT)
            raise

        tqdm.tqdm.write(f"Built {len(tracks)} tracks")

        if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 20:
            tqdm.tqdm.write(f"With {cfg.keys} => {values} tracking failed (too few or too many tracks). Continuing...")
            continue

        # Hota @ 2 (-8 => Thresholds is 2)
        hota = tracking_metrics.compute_tracking_metrics(tracks, ground_truth)["HOTA"][-8].item()

        tqdm.tqdm.write(f"With {cfg.keys} => {values},  HOTA@2.0: {hota}")

        if hota > best_hota:
            best_parameters = linker_parameters
            best_values = values
            best_hota = hota

    if best_parameters != parameters:
        print("The default configuration is sub-optimal, we found a better one:")
        print(f"{cfg.keys} => {best_values}, HOTA@2.0: {best_hota}")

    print("Best parameters:")
    print(yaml.dump(dataclasses.asdict(best_parameters)))

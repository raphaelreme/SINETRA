import glob
import pathlib
from typing import Dict, List

import numpy as np
import yaml  # type: ignore


def load_results(
    exp_folder: pathlib.Path, motions: List[str], tracking_methods: List[str], detection_methods: List[str]
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Find all the results in the exp folder for the given tracking methods and detections methods"""
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        tracking_method: {
            motion: {detection_method: [] for detection_method in detection_methods} for motion in motions
        }
        for tracking_method in tracking_methods
    }
    for motion in motions:
        for path in glob.glob(str(exp_folder / "tracking" / f"{motion}" / "1.5-50.0" / "*" / "*")):
            if not (pathlib.Path(path) / "metrics.yml").exists():
                continue  # Run has not finished

            detection_cfg = yaml.safe_load((pathlib.Path(path) / "config.yml").read_text())["detection"]

            detection_name = ""
            if detection_cfg["detector"] == "wavelet":
                detection_name = "Wavelet"
            elif detection_cfg["detector"] == "fake":
                if detection_cfg["fake"]["fpr"] != detection_cfg["fake"]["fnr"]:
                    print("Found unbalanced fake detections, skipping...")
                    continue
                detection_name = f"Fake@{int((1 - detection_cfg['fake']['fpr']) * 100)}%"

            if detection_name not in detection_methods:
                continue

            metrics = yaml.safe_load((pathlib.Path(path) / "metrics.yml").read_text(encoding="utf-8"))

            for tracking_method in tracking_methods:
                if tracking_method not in metrics:
                    continue

                results[tracking_method][motion][detection_name].append(metrics[tracking_method]["HOTA"])

    return results


def main():
    motions = ["hydra_flow", "springs"]
    detection_methods = ["Wavelet", "Fake@80%"]
    tracking_methods = {
        "trackmate": "U-Track",
        "emht": "eMHT",
        "koft": "KOFT",
        "zephir-low": "ZephIR@3",
        "zephir": "ZephIR@10",
    }

    # Get results
    results = load_results(pathlib.Path("experiment_folder"), motions, list(tracking_methods.keys()), detection_methods)

    det_n = max(max(len(det) for det in detection_methods), 20)
    trk_n = max(max(len(trk) for trk in tracking_methods.values()), 10)
    mtn_n = det_n * len(detection_methods) + len(detection_methods) - 1

    # Header:
    print("|".join([f"{'':^{trk_n}}"] + [f"{motion:^{mtn_n}}" for motion in motions]))
    print("-" * (mtn_n * len(motions) + trk_n + len(motions)))
    detection_line = [f"{'Detection':^{trk_n}}"]
    for motion in motions:
        for detection_name in detection_methods:
            detection_line.append(f"{detection_name:^{det_n}}")

    print("|".join(detection_line))
    print("-" * (mtn_n * len(motions) + trk_n + len(motions)))

    for tracking_method in tracking_methods:  # pylint: disable=consider-using-dict-items
        tracking_line = [f"{tracking_methods[tracking_method]:^{trk_n}}"]

        for motion in motions:
            for detection_name in detection_methods:
                scores = results[tracking_method][motion][detection_name]

                if scores:
                    mean, std = float(np.mean(scores)), float(np.std(scores, ddof=0))
                else:
                    mean, std = -1.0, 0.0

                results_str = f"{mean*100:.1f} +/- {std*100:0.1f}% ({len(scores)})"
                tracking_line.append(f"{results_str:^{det_n}}")

        print("|".join(tracking_line))


if __name__ == "__main__":
    main()

"""All tracking experiments in the dataset"""

import subprocess

config: str = "configs/tracking/track.yml"
seeds = [111, 222, 333, 444, 555]


# Wavelet detections
for data_name in ("springs", "hydra_flow"):
    for seed in seeds:
        subprocess.run(f"expyrun {config} --seed {seed} --data_name {data_name}", check=True, shell=True)


# Fake detections
for data_name in ("springs", "hydra_flow"):
    for seed in seeds:
        subprocess.run(
            f"expyrun {config} --seed {seed} --data_name {data_name} --detection.detector fake", check=True, shell=True
        )

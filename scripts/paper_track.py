"""All tracking experiments in the dataset"""

import subprocess

config: str = "configs/tracking/track.yml"
seeds = [111, 222, 333, 444, 555]


# 2D
for data_name in ("hydra_flow", "springs_2d"):
    for seed in seeds:
        subprocess.run(
            f"expyrun {config} --seed {seed} --data_name {data_name} --tracking_methods trackmate,emht,koft",
            check=True,
            shell=True,
        )
        # Do ZephIR on its own with larger detections (less good for a detection task, but zephir only uses the mask)
        subprocess.run(
            f"expyrun {config} --seed {seed} --data_name {data_name} --detection.wavelet.k 1.0"
            " --tracking_methods zephir,zephir-low",
            check=True,
            shell=True,
        )


# 3D -> Set wavelet k to 6.0 to be able to separate spots (cf grid_search_detect)
for seed in seeds:
    # For KOFT let's use tvl1 that supports 3D (but it does not work that good, so let's reduce the belief)
    subprocess.run(
        f"expyrun {config} --seed {seed} --data_name springs_3d  --detection.wavelet.k 6.0"
        " --tracking_methods trackmate,emht,koft --koft.optical_flow.name tvl1 --koft.flow_std 2.0",
        check=True,
        shell=True,
    )
    # Do ZephIR on its own with larger detections (less good for a detection task, but zephir only uses the mask)
    # Also in 3D, use grid_step=11 (better and much much faster) and lambda_n=0.01
    subprocess.run(
        f"expyrun {config} --seed {seed} --data_name springs_3d --detection.wavelet.k 4.0"
        " --tracking_methods zephir,zephir-low --zephir.grid_step 11 --zephir.lambda_n 0.01",
        check=True,
        shell=True,
    )

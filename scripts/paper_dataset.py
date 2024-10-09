"""Generate the dataset used in the paper"""

import subprocess


for seed in [111, 222, 333, 444, 555]:
    subprocess.run(f"expyrun configs/dataset/springs.yml --seed {seed}", check=True, shell=True)
    subprocess.run(f"expyrun configs/dataset/hydra_flow.yml --seed {seed}", check=True, shell=True)

import random

import numpy as np
import torch
import torch.backends.cudnn

# The imaging_genrator is unused... We currently use sample from torch.Distribution,
# which does not seem to support generators


particle_generator = torch.Generator()
imaging_generator = torch.Generator()
emission_generator = torch.Generator()


def enforce_all_seeds(seed: int, strict=True):
    """Enforce all the seeds

    If strict you may have to define the following env variable:
        CUBLAS_WORKSPACE_CONFIG=:4096:8  (Increase a bit the memory foot print ~25Mo)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.benchmark = False  # By default should already be to False
        torch.use_deterministic_algorithms(True, warn_only=True)

    particle_generator.manual_seed(seed + 1)
    imaging_generator.manual_seed(seed + 2)
    emission_generator.manual_seed(seed + 3)

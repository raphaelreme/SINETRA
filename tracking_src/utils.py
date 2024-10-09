import random

import numpy as np
import torch


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
        torch.use_deterministic_algorithms(True)

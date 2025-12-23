from __future__ import annotations

import os
import random

# numpy & related imports
import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("high")  # A100
    except Exception:
        pass  # torch is not installed

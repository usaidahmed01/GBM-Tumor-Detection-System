from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python inside DataLoader workers using PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

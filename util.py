import json
import random
import torch
import numpy as np

from typing import NamedTuple


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 42
    batch_size: int = 32
    lr: int = 2e-5
    epochs: int = 10
    warmup: float = 0.1
    save_steps: int = 100

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

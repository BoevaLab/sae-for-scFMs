"""
Utility functions and classes for SAE training and evaluation.
"""

import os
import scvi
import numpy as np
from scanpy.pp import subsample
import torch
import random
from torch.utils.data import Dataset
from typing import Dict


class OutputHook:
    """
    Forward hook to save or modify activations
    """
    def __init__(self):
        self.outputs = []
        self.x_hat = None
        self.modify = False

    def __call__(self, module, module_in, module_out):
        if self.modify:
            return self.x_hat
        else:
            self.outputs.append(module_out)
        return module_out

    def clear(self):
        self.outputs = []
    
    def __enter__(self):
        """Context manager entry: clear outputs before use"""
        self.clear()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: always clear outputs after use"""
        self.clear()
        return False  # Don't suppress exceptions

class SeqDataset(Dataset):
    """
    Custom Dataset for sequence data
    """
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data["genes"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
def get_devices() -> tuple:
    """
    Determines available CUDA devices and returns the primary device for training and a list of buffer devices.

    Returns:
        tuple: (device_training, device_buffer) where device_training is a string representing the main device,
               and device_buffer is a list of strings for additional devices (or ['cpu'] if no CUDA devices are available).
    """
    devices =  [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if len(devices) == 0:
        device_training, device_buffer = 'cpu', ['cpu']
    elif len(devices) == 1:
        device_training, device_buffer = devices[0], [devices[0]]
    else:
        device_training = devices[0]
        device_buffer = devices[1:]
    return (device_training, device_buffer)

def set_seed(seed) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

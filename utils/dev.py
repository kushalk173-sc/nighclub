import torch
import os

_device = None

def get_device():
    """
    Gets the singleton torch device.
    Defaults to CUDA if available, otherwise CPU.
    """
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Device: Singleton device set to '{_device}'.")
    return _device 
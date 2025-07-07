import torch
import os

# Global device singleton
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

def to_device(obj):
    """Move any tensor or tensor-like object to the correct device."""
    dev = get_device()
    if isinstance(obj, torch.Tensor):
        return obj.to(dev, non_blocking=True)
    elif hasattr(obj, 'to'):
        return obj.to(dev, non_blocking=True)
    else:
        return obj 
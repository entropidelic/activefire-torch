"""Choose where PyTorch runs: an NVIDIA GPU ("cuda"), an Apple GPU ("mps") or the CPU.

A machine has at most one of the two GPU kinds, so "auto" simply uses the GPU
if there is one, and the CPU otherwise.
"""

import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def get_device(choice: str = "auto") -> torch.device:
    """Return the device to run on.

    choice:
        "auto"                use the GPU if there is one, else the CPU.
        "cuda", "mps", "cpu"  use exactly that one. If it is not available,
                              raise an error instead of quietly training on
                              something slower.
    """
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {choice!r}; choose one of {DEVICE_CHOICES}")

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if choice == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda" and not cuda_available:
        raise RuntimeError("Device 'cuda' was asked for, but there is no NVIDIA GPU")
    if choice == "mps" and not mps_available:
        raise RuntimeError("Device 'mps' was asked for, but there is no Apple GPU")
    return torch.device(choice)


def describe_device(device: torch.device) -> str:
    """A short description of the device, saved with each training run."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type

"""Compute device selection.

The order is CUDA -> MPS -> CPU. CUDA and MPS never coexist on one machine
(NVIDIA vs. Apple Silicon), so in practice this picks whichever GPU backend
exists and only falls back to CPU when there is none.
"""

import torch

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def get_device(preference: str = "auto") -> torch.device:
    """Return the torch device to use.

    Args:
        preference: "auto" picks the best available backend. "cuda", "mps" or
            "cpu" force that backend and fail loudly if it is unavailable,
            instead of silently training somewhere slower.
    """
    if preference not in VALID_DEVICES:
        raise ValueError(f"Unknown device {preference!r}; expected one of {VALID_DEVICES}")

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' requested but MPS is not available")
    return torch.device(preference)


def describe_device(device: torch.device) -> str:
    """Human-readable device description, saved with each run's config."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type

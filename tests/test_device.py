import pytest
import torch

from activefire_torch.device import get_device


def test_auto_returns_an_available_device():
    device = get_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=device)  # the device is actually usable


def test_auto_prefers_gpu_over_cpu():
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if gpu_available:
        assert get_device("auto").type != "cpu"


def test_cpu_is_always_available():
    assert get_device("cpu").type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_forcing_unavailable_cuda_raises():
    with pytest.raises(RuntimeError):
        get_device("cuda")

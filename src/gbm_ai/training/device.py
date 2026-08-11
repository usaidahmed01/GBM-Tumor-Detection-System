from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if requested == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise RuntimeError("MPS requested but it is not available.")

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    return torch.device(requested)


def describe_device(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info

"""Backward-compatible imports for runtime helpers."""

from .core.runtime import (
    detected_torch_device,
    inference_torch_dtype,
    runtime_device,
)

__all__ = [
    "detected_torch_device",
    "inference_torch_dtype",
    "runtime_device",
]

"""giftpose — self-contained replacement for mmpose.apis.MMPoseInferencer.

The drop-in entry point is ``libs.giftpose.MMPoseInferencer``. It is imported
lazily so submodule-level CLIs (``python -m libs.giftpose.export.onnx_export``,
etc.) work without the full dependency chain loading on every import.
"""
from __future__ import annotations

__all__ = ["MMPoseInferencer"]


def __getattr__(name: str):  # PEP 562 lazy import
    if name == "MMPoseInferencer":
        from libs.giftpose.inferencer import MMPoseInferencer as _Inf
        return _Inf
    raise AttributeError(f"module 'libs.giftpose' has no attribute {name!r}")

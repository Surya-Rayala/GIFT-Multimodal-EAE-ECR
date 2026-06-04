"""Build TensorRT engines from exported ONNX files via the TensorRT Python API.

The TRT runtime backend (libs/giftpose/runtime/trt_backend.py) consumes the
generated ``*.engine`` files. Requires ``pip install tensorrt`` (the Python
bindings — ``trtexec`` is intentionally not used so the pip wheel alone is
sufficient; NVIDIA's TensorRT tarball/deb is not required). Install reference:
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html

Usage (one-shot — auto-exports ONNX from .pth if not already present):
    python -m libs.giftpose.export.trt_build

Or with explicit paths:
    python -m libs.giftpose.export.trt_build \
        --det-weights models/detect-best-mAP.pth --det-engine models/detect-best-mAP.engine \
        --pose-weights models/pose.pth --pose-engine models/pose.engine
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from libs.giftpose.export.onnx_export import (
    export_onnx_detector,
    export_onnx_pose,
)


def _parse_shape(spec: str) -> Tuple[str, Tuple[int, ...]]:
    """Parse ``images:1x3x384x288`` -> ``("images", (1, 3, 384, 288))``."""
    name, dims = spec.split(":", 1)
    return name, tuple(int(d) for d in dims.split("x"))


def _ensure_onnx(weights_path: str, onnx_path: str, kind: str) -> None:
    """Export ONNX from .pth if the .onnx artifact is missing."""
    if Path(onnx_path).exists():
        return
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"{kind} weights not found at {weights_path}; cannot auto-export ONNX."
        )
    print(f"[trt_build] {onnx_path} missing — exporting from {weights_path}")
    if kind == "detector":
        export_onnx_detector(weights_path, onnx_path)
    elif kind == "pose":
        export_onnx_pose(weights_path, onnx_path)
    else:
        raise ValueError(f"unknown kind: {kind}")


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    min_shape: str | None = None,
    opt_shape: str | None = None,
    max_shape: str | None = None,
    workspace_mb: int = 4096,
) -> None:
    import tensorrt as trt  # type: ignore

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Reuse layer-timing measurements across builds — second + subsequent
    # builds for the same GPU/TRT pair are 2-5x faster.
    cache_path = Path(engine_path).parent / ".trt_timing_cache"
    cache_data = cache_path.read_bytes() if cache_path.exists() else b""
    timing_cache = config.create_timing_cache(cache_data)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # Dynamic shapes are supplied via an optimization profile.
    if min_shape and opt_shape and max_shape:
        profile = builder.create_optimization_profile()
        name_min, dims_min = _parse_shape(min_shape)
        name_opt, dims_opt = _parse_shape(opt_shape)
        name_max, dims_max = _parse_shape(max_shape)
        if not (name_min == name_opt == name_max):
            raise ValueError(
                f"min/opt/max shape names disagree: {name_min}/{name_opt}/{name_max}"
            )
        profile.set_shape(name_min, dims_min, dims_opt, dims_max)
        config.add_optimization_profile(profile)

    print(f"building {engine_path} (fp16={fp16})")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine for {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    cache_path.write_bytes(memoryview(timing_cache.serialize()).tobytes())
    print(f"built {engine_path}")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--det-onnx", default="models/detect-best-mAP.onnx")
    p.add_argument("--det-engine", default="models/detect-best-mAP.engine")
    p.add_argument("--pose-onnx", default="models/pose.onnx")
    p.add_argument("--pose-engine", default="models/pose.engine")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    args = p.parse_args(argv)

    fp16 = not args.no_fp16
    if not args.skip_detector:
        _ensure_onnx(args.det_weights, args.det_onnx, "detector")
        build_engine(args.det_onnx, args.det_engine, fp16=fp16,
                     workspace_mb=args.workspace_mb)
    if not args.skip_pose:
        _ensure_onnx(args.pose_weights, args.pose_onnx, "pose")
        # Pose handles a variable number of detections per frame; the TRT
        # engine carries dynamic batch range 1..32 (opt at 8).
        build_engine(
            args.pose_onnx, args.pose_engine, fp16=fp16,
            min_shape="images:1x3x384x288",
            opt_shape="images:8x3x384x288",
            max_shape="images:32x3x384x288",
            workspace_mb=args.workspace_mb,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

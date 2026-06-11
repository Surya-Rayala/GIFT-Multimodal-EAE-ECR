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
    verify_onnx,
)


def _parse_shape(spec: str) -> Tuple[str, Tuple[int, ...]]:
    """Parse ``images:1x3x384x288`` -> ``("images", (1, 3, 384, 288))``."""
    name, dims = spec.split(":", 1)
    return name, tuple(int(d) for d in dims.split("x"))


def _make_builder(trt, logger):
    """Create a ``trt.Builder``, turning the opaque pybind11 nullptr failure
    (GPU below TensorRT's SM floor, or driver older than the wheel's CUDA) into
    an actionable message."""
    try:
        builder = trt.Builder(logger)
    except Exception as e:
        hint = ""
        try:
            import torch
            cap = torch.cuda.get_device_capability()
            hint = (f" Detected GPU '{torch.cuda.get_device_name()}' "
                    f"(compute capability sm_{cap[0]}{cap[1]}).")
        except Exception:
            pass
        raise RuntimeError(
            f"TensorRT {getattr(trt, '__version__', '?')} could not create a Builder on "
            f"this machine.{hint}\n"
            f"TensorRT 10 supports only compute capability >= 7.5 (Turing or newer); "
            f"Pascal (sm_61) and older GPUs are unsupported. This also fires when the "
            f"NVIDIA driver is older than the installed wheel's CUDA.\n"
            f"If your GPU is pre-Turing, skip the TRT backend and use ONNX Runtime — the "
            f"exported .onnx is auto-selected by libs.giftpose (TRT > ONNX Runtime > "
            f"PyTorch), so no engine is needed:\n"
            f"    pip install --force-reinstall onnxruntime-gpu\n"
            f"For an older GPU that must use TensorRT, install TensorRT 8.6.x (the last "
            f"line supporting Pascal)."
        ) from e
    if builder is None:  # defensive: some builds return None rather than raising
        raise RuntimeError("trt.Builder returned None — see the TensorRT log above.")
    return builder


def _onnx_to_fp16(onnx_path: str | Path) -> Path | None:
    """Convert an fp32 ONNX to fp16 (``keep_io_types`` so I/O stays fp32) for
    TensorRT *strong typing* (TRT 10.12+ dropped ``BuilderFlag.FP16``). Returns
    the new path, or ``None`` when conversion isn't possible — the caller then
    builds a correct fp32 engine."""
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        print(f"[trt_build] strong-typed fp16 needs onnxconverter-common "
              f"(`pip install onnxconverter-common`); import failed: {e}. "
              f"Building an fp32 engine instead.")
        return None
    try:
        model16 = float16.convert_float_to_float16(
            onnx.load(str(onnx_path)), keep_io_types=True
        )
        out = Path(onnx_path).with_name(Path(onnx_path).stem + "_fp16.onnx")
        onnx.save(model16, str(out))
        return out
    except Exception as e:  # model-specific
        print(f"[trt_build] fp16 onnx conversion failed ({e}); building fp32 engine.")
        return None


def _ensure_onnx(weights_path: str, onnx_path: str, kind: str, verify: bool = True) -> None:
    """Export ONNX from .pth if the .onnx artifact is missing, then verify it
    matches the eager model so a stale or buggy ONNX can't silently become a
    TRT engine.

    The verify step runs whether the ONNX was just exported OR already on disk —
    a pre-existing ``.onnx`` left over from an older checkpoint is the classic
    cause of "PyTorch works, TRT is wrong". Verification needs onnxruntime; if
    it's absent in the build env we warn loudly rather than fail (the engine is
    still built, just unverified)."""
    if not Path(onnx_path).exists():
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

    if verify:
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            print(f"[trt_build] WARNING: onnxruntime not installed — skipping "
                  f"{kind} ONNX verification. Install onnxruntime to guard "
                  f"against stale/mismatched {onnx_path}.")
            return
        # Raises if the ONNX diverges from eager (stale checkpoint or bad export).
        verify_onnx(onnx_path, weights_path, kind)


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
    builder = _make_builder(trt, logger)

    # Pick the FP16 route by what THIS TensorRT exposes, so fp16 perf works on
    # every version:
    #   * weak typing (TRT < 10.12): build from the fp32 onnx + BuilderFlag.FP16.
    #   * strong typing (TRT 10.12+ dropped the flag): a STRONGLY_TYPED network
    #     built from an fp16 onnx (keep_io_types -> fp32 I/O, fp16 compute).
    # If neither is possible we build a correct fp32 engine.
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    st_flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    eb_flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)

    use_strong = bool(fp16 and fp16_flag is None and st_flag is not None)
    onnx_for_build: str | Path = onnx_path
    if use_strong:
        fp16_onnx = _onnx_to_fp16(onnx_path)
        if fp16_onnx is not None:
            onnx_for_build = fp16_onnx
        else:
            use_strong = False  # couldn't make an fp16 onnx -> fp32 engine

    if use_strong:
        network = builder.create_network(1 << int(st_flag))  # explicit batch implied
    else:
        network = builder.create_network(
            1 << int(eb_flag) if eb_flag is not None else 0
        )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_for_build, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed for {onnx_for_build}: {errs}")

    config = builder.create_builder_config()
    # Workspace limit: set_memory_pool_limit is TRT 8.4+/10; older builds use
    # the deprecated max_workspace_size attribute.
    _workspace = workspace_mb * (1 << 20)
    _pool = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and _pool is not None:
        config.set_memory_pool_limit(_pool.WORKSPACE, _workspace)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = _workspace
    # Weak-typing fp16 flag (TRT < 10.12). Strong-typed engines carry fp16 in
    # the onnx itself, so no flag is set for them.
    if (fp16 and not use_strong and fp16_flag is not None
            and getattr(builder, "platform_has_fast_fp16", True)):
        config.set_flag(fp16_flag)

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

    eff_fp16 = use_strong or (fp16 and fp16_flag is not None)
    print(f"building {engine_path} (precision={'fp16' if eff_fp16 else 'fp32'})")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            f"TensorRT failed to build engine for {onnx_for_build} (returned None — "
            f"see the TensorRT log above; usually a driver/CUDA mismatch or an "
            f"unsupported op)."
        )
    with open(engine_path, "wb") as f:
        # IHostMemory -> bytes is portable across TRT versions.
        f.write(bytes(serialized))
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
    p.add_argument("--no-fp16", action="store_true",
                   help="Force fp32 for BOTH detector and pose engines.")
    p.add_argument("--det-fp16", action="store_true",
                   help="Build the detector in fp16 too (default fp32 — its "
                        "scores ride the 0.5 confidence gate and fp16 flips "
                        "boundary detections, causing missing/flickering boxes).")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the eager-vs-ONNX parity check before building "
                        "(NOT recommended — verification is what catches a stale "
                        "or buggy .onnx becoming a wrong engine).")
    args = p.parse_args(argv)
    verify = not args.no_verify

    # Precision is decoupled per model. The RTMDet person-detector's sigmoid
    # scores cluster tightly around the 0.5 confidence gate applied downstream
    # (``processing_engine.box_conf_threshold`` -> ``inferencer`` ``bbox_thr``),
    # so fp16 accumulation through the backbone flips boundary detections in and
    # out frame-to-frame — the "missing / inconsistent detections" symptom that
    # PyTorch / TorchScript (fp32) don't show. The detector is a single 640x640
    # pass, so building it fp32 is cheap and keeps the gate identical to the
    # reference backends. Pose stays fp16 (keypoint argmax shift is sub-pixel).
    # ``--no-fp16`` forces both to fp32; ``--det-fp16`` opts the detector back
    # into fp16 for users who accept the accuracy risk for speed.
    det_fp16 = (not args.no_fp16) and args.det_fp16
    pose_fp16 = not args.no_fp16
    if not args.skip_detector:
        _ensure_onnx(args.det_weights, args.det_onnx, "detector", verify=verify)
        build_engine(args.det_onnx, args.det_engine, fp16=det_fp16,
                     workspace_mb=args.workspace_mb)
    if not args.skip_pose:
        _ensure_onnx(args.pose_weights, args.pose_onnx, "pose", verify=verify)
        # Pose handles a variable number of detections per frame; the TRT
        # engine carries dynamic batch range 1..32 (opt at 8).
        build_engine(
            args.pose_onnx, args.pose_engine, fp16=pose_fp16,
            min_shape="images:1x3x384x288",
            opt_shape="images:8x3x384x288",
            max_shape="images:32x3x384x288",
            workspace_mb=args.workspace_mb,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

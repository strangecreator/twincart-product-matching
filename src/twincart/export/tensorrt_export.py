from __future__ import annotations

import re
import json
import pathlib
import typing as tp
from dataclasses import dataclass


@dataclass(frozen=True)
class TrtPaths:
    image_engine: pathlib.Path | None
    text_engine: pathlib.Path | None


def _find_fold_name(path: pathlib.Path) -> str:
    for parent in path.parents:
        if re.match(r"fold_\d+", parent.name):
            return parent.name

    return "fold_0"


def _read_json(path: pathlib.Path) -> dict[str, tp.Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _onnx_meta(onnx_path: pathlib.Path) -> dict[str, tp.Any]:
    meta_path = onnx_path.parent / "meta.json"

    if meta_path.exists():
        return _read_json(meta_path)

    return {}


def _set_workspace_bytes(config: tp.Any, workspace_bytes: int) -> None:
    # trt <10
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = int(workspace_bytes)
        return

    # trt >=10
    import tensorrt as trt

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))


def _maybe_load_timing_cache(config: tp.Any, timing_cache: pathlib.Path) -> tp.Any | None:
    if not timing_cache.exists():
        return None

    try:
        blob = timing_cache.read_bytes()
        cache = config.create_timing_cache(blob)
        is_ok = config.set_timing_cache(cache, ignore_mismatch=False)

        return cache if is_ok else None
    except Exception:
        return None


def _maybe_save_timing_cache(config: tp.Any, timing_cache: pathlib.Path) -> None:
    try:
        cache = config.get_timing_cache()

        if cache is None:
            return

        timing_cache.parent.mkdir(parents=True, exist_ok=True)
        timing_cache.write_bytes(bytearray(cache.serialize()))
    except Exception:
        return


def build_engine_from_onnx(
    *,
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    precision: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_gb: float,
    timing_cache: pathlib.Path | None,
    force: bool,
) -> pathlib.Path:
    if engine_path.exists() and not force:
        return engine_path

    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError("TensorRT python bindings are not installed.") from e

    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported precision={precision!r}. Use 'fp32' or 'fp16'.")

    meta = _onnx_meta(onnx_path)

    is_image = ("h" in meta) and ("w" in meta)
    is_text = "max_len" in meta

    if not (is_image or is_text):
        raise RuntimeError(
            f"Could not infer input shapes for {onnx_path}.\n" "Expected meta.json next to ONNX with either keys {h, w} (image) or {max_len} (text)."
        )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()

    if not parser.parse(onnx_bytes):
        errors = []

        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))

        raise RuntimeError("Failed to parse ONNX with TensorRT:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    _set_workspace_bytes(config, int(workspace_gb * (1024**3)))

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            pass

    # optimization profile
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        name = network.get_input(i).name

        if is_image:
            h, w = int(meta["h"]), int(meta["w"])

            if name != "input":
                raise RuntimeError(f"Unexpected image input name {name!r} (expected 'input').")

            min_shape = (int(min_batch), 3, h, w)
            opt_shape = (int(opt_batch), 3, h, w)
            max_shape = (int(max_batch), 3, h, w)
            profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

        elif is_text:
            max_len = int(meta["max_len"])

            if name not in {"input_ids", "attention_mask"}:
                raise RuntimeError(f"Unexpected text input name {name!r} (expected 'input_ids' or 'attention_mask').")

            min_shape = (int(min_batch), max_len)
            opt_shape = (int(opt_batch), max_len)
            max_shape = (int(max_batch), max_len)
            profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

    config.add_optimization_profile(profile)

    # timing cache
    if timing_cache is not None:
        _maybe_load_timing_cache(config, timing_cache)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None (engine build failed).")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytearray(serialized))

    if timing_cache is not None:
        _maybe_save_timing_cache(config, timing_cache)

    # writing engine metadata
    out_meta = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": precision,
        "profile": {"min_batch": min_batch, "opt_batch": opt_batch, "max_batch": max_batch},
        "workspace_gb": workspace_gb,
        "tensorrt_version": getattr(trt, "__version__", "unknown"),
        "onnx_meta": meta,
    }
    (engine_path.parent / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    return engine_path


def export_trt_engines(
    *,
    image_onnx: pathlib.Path | None,
    text_onnx: pathlib.Path | None,
    out_dir: pathlib.Path,
    precision: str = "fp16",
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 64,
    workspace_gb: float = 4.0,
    timing_cache: pathlib.Path | None = None,
    force: bool = False,
) -> TrtPaths:
    image_engine = None
    text_engine = None

    if image_onnx is not None:
        fold = _find_fold_name(image_onnx)
        image_engine = out_dir / "image" / fold / "model.engine"
        build_engine_from_onnx(
            onnx_path=image_onnx,
            engine_path=image_engine,
            precision=precision,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            workspace_gb=workspace_gb,
            timing_cache=timing_cache,
            force=force,
        )

    if text_onnx is not None:
        fold = _find_fold_name(text_onnx)
        text_engine = out_dir / "text" / fold / "model.engine"
        build_engine_from_onnx(
            onnx_path=text_onnx,
            engine_path=text_engine,
            precision=precision,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            workspace_gb=workspace_gb,
            timing_cache=timing_cache,
            force=force,
        )

    return TrtPaths(image_engine=image_engine, text_engine=text_engine)

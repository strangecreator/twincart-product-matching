from __future__ import annotations

import re
import json
import pathlib
import inspect
import typing as tp
from dataclasses import dataclass

# hydra & related imports
from omegaconf import OmegaConf

# torch & related imports
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExportPaths:
    image_onnx: pathlib.Path | None
    text_onnx: pathlib.Path | None


def _find_fold_name(path: pathlib.Path) -> str:
    for parent in path.parents:
        if re.match(r"fold_\d+", parent.name):
            return parent.name

    return "fold_0"


def _load_resolved_cfg(ckpt_path: pathlib.Path) -> tp.Any | None:
    config_path = ckpt_path.parent / "resolved_config.yaml"

    if config_path.exists():
        return OmegaConf.load(config_path)

    return None


def _extract_encoder_state_dict(ckpt_path: pathlib.Path, prefix: str = "encoder.") -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)

    out: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value

    if not out:
        raise RuntimeError(f"Could not find any keys with prefix '{prefix}' in checkpoint: {ckpt_path}.")

    return out


def _build_image_encoder(config: tp.Any) -> nn.Module:
    from twincart.models.image_encoder import ImageEncoder

    image_model = config.get("image_model") if config is not None else None

    if image_model is None:
        raise RuntimeError("resolved_config.yaml does not contain 'image_model' section.")

    pooling = image_model.get("pooling", {})
    p_infer = float(pooling.get("p_infer", pooling.get("p_train", 3.0)))
    eps = float(pooling.get("eps", 1e-6))

    kwargs = dict(
        backbone=str(image_model.backbone),
        pretrained=False,  # weights come from ckpt
        embedding_dim=int(image_model.embedding_dim),
        gem_p=p_infer,
        gem_eps=eps,
    )

    if "weights_cfg" in inspect.signature(ImageEncoder.__init__).parameters:
        kwargs["weights_cfg"] = None

    return ImageEncoder(**kwargs)


def _build_text_encoder(config: tp.Any) -> nn.Module:
    from twincart.models.text_encoder import TextEncoder

    text_model = config.get("text_model") if config is not None else None

    if text_model is None:
        raise RuntimeError("resolved_config.yaml does not contain 'text_model' section.")

    return TextEncoder(
        backbone=str(text_model.backbone),
        embedding_dim=int(text_model.embedding_dim),
    )


class _TextOnnxWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


def export_image_to_onnx(
    ckpt_path: pathlib.Path,
    out_path: pathlib.Path,
    h: int,
    w: int,
    opset: int = 18,
    dynamo: bool = False,
) -> None:
    config = _load_resolved_cfg(ckpt_path)
    encoder = _build_image_encoder(config).cpu().eval()
    encoder.load_state_dict(_extract_encoder_state_dict(ckpt_path), strict=False)

    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["image_embedding"],
        dynamic_axes={
            "input": {0: "batch"},
            "image_embedding": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=dynamo,
    )

    meta = {
        "ckpt": str(ckpt_path),
        "h": h,
        "w": w,
        "opset": opset,
        "resolved_config": str(ckpt_path.parent / "resolved_config.yaml"),
    }
    (out_path.parent / "meta.json").write_text(json.dumps(meta, indent=2))


def export_text_to_onnx(
    ckpt_path: pathlib.Path,
    out_path: pathlib.Path,
    max_len: int,
    opset: int = 17,
    dynamo: bool = False,
) -> None:
    config = _load_resolved_cfg(ckpt_path)
    encoder = _build_text_encoder(config).cpu().eval()
    encoder.load_state_dict(_extract_encoder_state_dict(ckpt_path), strict=False)

    wrapped = _TextOnnxWrapper(encoder).cpu().eval()

    input_ids = torch.zeros(1, max_len, dtype=torch.long)
    attention_mask = torch.ones(1, max_len, dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(out_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_embedding": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=dynamo,
    )

    meta = {
        "ckpt": str(ckpt_path),
        "max_len": max_len,
        "opset": opset,
        "resolved_config": str(ckpt_path.parent / "resolved_config.yaml"),
    }
    (out_path.parent / "meta.json").write_text(json.dumps(meta, indent=2))


def export_both(
    image_ckpt: pathlib.Path | None,
    text_ckpt: pathlib.Path | None,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    opset: int,
    dynamo: bool,
) -> ExportPaths:
    image_out = None
    text_out = None

    if image_ckpt is not None:
        fold = _find_fold_name(image_ckpt)
        image_out = out_dir / "image" / fold / "model.onnx"
        export_image_to_onnx(image_ckpt, image_out, h=image_h, w=image_w, opset=opset, dynamo=dynamo)

    if text_ckpt is not None:
        fold = _find_fold_name(text_ckpt)
        text_out = out_dir / "text" / fold / "model.onnx"
        export_text_to_onnx(text_ckpt, text_out, max_len=text_max_len, opset=opset, dynamo=dynamo)

    return ExportPaths(image_onnx=image_out, text_onnx=text_out)

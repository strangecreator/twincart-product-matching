from __future__ import annotations

import os
import io
import base64
import pathlib
import typing as tp
from dataclasses import dataclass

# utility imports
import numpy as np
import pandas as pd
from PIL import Image

# mlflow & related imports
import mlflow.pyfunc


@dataclass
class PreprocessCfg:
    image_h: int = 420
    image_w: int = 420
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    text_max_len: int = 64


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _resolve_onnx_artifact_path(path: str) -> str:
    """
    MLflow artifacts can be:
      - a file path (for example, ".../artifacts/image_onnx")
      - a directory path (for example, ".../artifacts/image_onnx/") containing "*.onnx"

    Returns the actual ONNX file path.
    """

    path = pathlib.Path(path)

    if path.is_file():
        return str(path)

    if path.is_dir():
        candidate = path / "model.onnx"
        if candidate.exists():
            return str(candidate)

        onnx_files = sorted(path.glob("*.onnx"))
        if len(onnx_files) == 1:
            return str(onnx_files[0])

        raise RuntimeError(f"Artifact directory {path} must contain exactly one *.onnx or a model.onnx, found: {onnx_files}.")

    raise RuntimeError(f"Artifact path does not exist: {path}.")


def _load_rgb_pil(image_path: str | None, image_b64: str | None) -> Image.Image:
    if image_b64 is not None and isinstance(image_b64, str) and len(image_b64) > 0:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if image_path is None or not isinstance(image_path, str) or len(image_path) == 0:
        raise ValueError("Either 'image_path' or 'image_b64' must be provided per row.")

    return Image.open(image_path).convert("RGB")


def _preprocess_image_batch(
    image_paths: list[str | None],
    image_b64s: list[str | None],
    config: PreprocessCfg,
) -> np.ndarray:
    mean = np.asarray(config.image_mean, dtype=np.float32)[None, None, :]  # (1, 1, 3)
    std = np.asarray(config.image_std, dtype=np.float32)[None, None, :]  # (1, 1, 3)

    batch = []
    for path, b64 in zip(image_paths, image_b64s, strict=True):
        im = _load_rgb_pil(path, b64)
        im = im.resize((config.image_w, config.image_h), resample=Image.BILINEAR)

        array = np.asarray(im, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected image array (h, w, 3), got shape={array.shape}.")

        batch.append(np.transpose((array - mean) / std, (2, 0, 1)))  # (3, h, w)

    return np.stack(batch, axis=0).astype(np.float32)  # (B, 3, h, w)


def _input_names(session) -> list[str]:
    return [input_info.name for input_info in session.get_inputs()]


def _output_names(session) -> list[str]:
    return [output_info.name for output_info in session.get_outputs()]


def _has_text_inputs(session) -> bool:
    input_name_set = set(_input_names(session))
    return ("input_ids" in input_name_set) or ("attention_mask" in input_name_set)


def _has_4d_input(session) -> bool:
    for input_info in session.get_inputs():
        shape = input_info.shape

        try:
            if shape is not None and len(shape) == 4:
                return True
        except Exception:
            pass

    return False


class TwinCartOnnxEncoderPyFunc(mlflow.pyfunc.PythonModel):
    """
    Input DataFrame:
      - title (required)
      - image_path (optional if image_b64 provided)
      - image_b64 (optional if image_path provided)

    Output DataFrame:
      - image_embedding, text_embedding, embedding (concatenate + normalize)
    """

    def __init__(self, config: PreprocessCfg, tokenizer_name_or_path: str) -> None:
        super().__init__()

        self.config = config
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self._img_sess = None
        self._txt_sess = None
        self._tokenizer = None

        self._img_in = None
        self._img_out = None

        self._txt_out = None
        self._txt_in_ids = None
        self._txt_in_mask = None
        self._txt_in_type_ids = None

    def load_context(self, context):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        available = ort.get_available_providers()
        providers: list[str] = []

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        img_onnx_path = _resolve_onnx_artifact_path(context.artifacts["image_onnx"])
        txt_onnx_path = _resolve_onnx_artifact_path(context.artifacts["text_onnx"])

        s1 = ort.InferenceSession(img_onnx_path, sess_options=so, providers=providers)
        s2 = ort.InferenceSession(txt_onnx_path, sess_options=so, providers=providers)

        # validation
        s1_img = _has_4d_input(s1) and (not _has_text_inputs(s1))
        s2_img = _has_4d_input(s2) and (not _has_text_inputs(s2))

        s1_txt = _has_text_inputs(s1)
        s2_txt = _has_text_inputs(s2)

        if s1_img and s2_txt and not s2_img:
            img_sess, txt_sess = s1, s2
        elif s2_img and s1_txt and not s1_img:
            img_sess, txt_sess = s2, s1
        else:
            raise RuntimeError(
                "MLflow model artifacts are wrong (likely overwritten during save_model).\n"
                f"image_onnx resolved to: {img_onnx_path}\n"
                f"text_onnx resolved to: {txt_onnx_path}\n"
                f"session1 inputs={_input_names(s1)} outputs={_output_names(s1)}\n"
                f"session2 inputs={_input_names(s2)} outputs={_output_names(s2)}\n"
                "Expected: image has 4D input and no input_ids.\n"
                "Expected: text has input_ids (and attention_mask).\n"
            )

        self._img_sess = img_sess
        self._txt_sess = txt_sess

        # tokenizer
        tokenizer_path = context.artifacts.get("tokenizer")
        if tokenizer_path and os.path.isdir(tokenizer_path):
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)

        # image i/o
        self._img_in = self._img_sess.get_inputs()[0].name
        self._img_out = self._img_sess.get_outputs()[0].name

        # text i/o
        txt_in_names = _input_names(self._txt_sess)
        txt_set = set(txt_in_names)

        self._txt_out = self._txt_sess.get_outputs()[0].name
        self._txt_in_ids = "input_ids" if "input_ids" in txt_set else txt_in_names[0]
        self._txt_in_mask = "attention_mask" if "attention_mask" in txt_set else None
        self._txt_in_type_ids = "token_type_ids" if "token_type_ids" in txt_set else None

    def predict(self, context, model_input: tp.Any) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("MLflow pyfunc expects pandas.DataFrame input")

        if len(model_input) == 0:
            return pd.DataFrame({"image_embedding": [], "text_embedding": [], "embedding": []})

        if "title" not in model_input.columns:
            raise ValueError("Input must contain 'title' column")

        if self._img_sess is None or self._txt_sess is None or self._tokenizer is None:
            raise RuntimeError("Model context is not loaded")

        titles = model_input["title"].astype(str).tolist()
        image_paths = model_input["image_path"].tolist() if "image_path" in model_input.columns else [None] * len(titles)
        image_b64s = model_input["image_b64"].tolist() if "image_b64" in model_input.columns else [None] * len(titles)

        # image
        x_img = _preprocess_image_batch(image_paths, image_b64s, self.config)
        img_embedding = _l2_normalize(self._img_sess.run([self._img_out], {self._img_in: x_img})[0].astype(np.float32))

        # text
        tokenizer = self._tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=self.config.text_max_len,
            return_tensors="np",
        )

        feed: dict[str, np.ndarray] = {self._txt_in_ids: tokenizer["input_ids"].astype(np.int64)}

        if self._txt_in_mask is not None:
            feed[self._txt_in_mask] = tokenizer["attention_mask"].astype(np.int64)

        if self._txt_in_type_ids is not None:
            if "token_type_ids" in tokenizer:
                feed[self._txt_in_type_ids] = tokenizer["token_type_ids"].astype(np.int64)
            else:
                feed[self._txt_in_type_ids] = np.zeros_like(tokenizer["input_ids"], dtype=np.int64)

        txt_embedding = _l2_normalize(self._txt_sess.run([self._txt_out], feed)[0].astype(np.float32))

        # image + text
        embedding = _l2_normalize(np.concatenate([img_embedding, txt_embedding], axis=1))

        return pd.DataFrame(
            {
                "image_embedding": [v.tolist() for v in img_embedding],
                "text_embedding": [v.tolist() for v in txt_embedding],
                "embedding": [v.tolist() for v in embedding],
            }
        )

from __future__ import annotations

import shutil
import pathlib
import tempfile

# click & related imports
import click


def _rm_rf(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _copy_onnx_bundle(src_onnx: pathlib.Path, dst_dir: pathlib.Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    # main onnx
    shutil.copy2(src_onnx, dst_dir / "model.onnx")

    # external weights
    data_path = src_onnx.with_suffix(".onnx.data")
    if data_path.exists():
        shutil.copy2(data_path, dst_dir / data_path.name)

    # metadata
    meta_path = src_onnx.parent / "meta.json"
    if meta_path.exists():
        shutil.copy2(meta_path, dst_dir / meta_path.name)


def build_mlflow_model(
    image_onnx: pathlib.Path,
    text_onnx: pathlib.Path,
    tokenizer_name: str,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    force: bool,
) -> None:
    if force:
        _rm_rf(out_dir)

    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    import mlflow
    from transformers import AutoTokenizer

    from twincart.serving.onnx_pyfunc import PreprocessCfg, TwinCartOnnxEncoderPyFunc

    with tempfile.TemporaryDirectory(prefix="twincart-mlflow-stage-") as temp_dir:
        stage = pathlib.Path(temp_dir)

        stage_image_path = stage / "image_onnx"
        stage_text_path = stage / "text_onnx"
        stage_tokenizer_path = stage / "tokenizer"

        stage_image_path.mkdir(parents=True, exist_ok=True)
        stage_text_path.mkdir(parents=True, exist_ok=True)
        stage_tokenizer_path.mkdir(parents=True, exist_ok=True)

        _copy_onnx_bundle(image_onnx, stage_image_path)
        _copy_onnx_bundle(text_onnx, stage_text_path)

        AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True).save_pretrained(stage_tokenizer_path)

        config = PreprocessCfg(image_h=image_h, image_w=image_w, text_max_len=text_max_len)
        py_model = TwinCartOnnxEncoderPyFunc(config=config, tokenizer_name_or_path=tokenizer_name)

        mlflow.pyfunc.save_model(
            path=str(out_dir),
            python_model=py_model,
            artifacts={
                "image_onnx": str(stage_image_path),
                "text_onnx": str(stage_text_path),
                "tokenizer": str(stage_tokenizer_path),
            },
            pip_requirements=None,
        )


@click.command(context_settings={"show_default": True})
@click.option("--image-onnx", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option("--text-onnx", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option("--tokenizer", "tokenizer_name", type=str, required=True, help="HF repo id or local path.")
@click.option("--out-dir", type=click.Path(path_type=pathlib.Path, file_okay=False), required=True)
@click.option("--image-h", type=int, default=420)
@click.option("--image-w", type=int, default=420)
@click.option("--text-max-len", type=int, default=64)
@click.option("--force", is_flag=True, help="Overwrite out-dir if exists.")
def main(
    image_onnx: pathlib.Path,
    text_onnx: pathlib.Path,
    tokenizer_name: str,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    force: bool,
) -> None:
    build_mlflow_model(
        image_onnx=image_onnx,
        text_onnx=text_onnx,
        tokenizer_name=tokenizer_name,
        out_dir=out_dir,
        image_h=image_h,
        image_w=image_w,
        text_max_len=text_max_len,
        force=force,
    )
    click.echo(f"Saved MLflow model to: {out_dir.resolve()}.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import typing as tp

# click & related imports
import click


def export_onnx_models(
    image_ckpt: pathlib.Path | None,
    text_ckpt: pathlib.Path | None,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    opset: int,
    dynamo: bool,
) -> tp.Any:
    image_ckpt = image_ckpt.resolve() if image_ckpt else None
    text_ckpt = text_ckpt.resolve() if text_ckpt else None
    out_dir = out_dir.resolve()

    from twincart.export.onnx_export import export_both

    return export_both(
        image_ckpt=image_ckpt,
        text_ckpt=text_ckpt,
        out_dir=out_dir,
        image_h=image_h,
        image_w=image_w,
        text_max_len=text_max_len,
        opset=opset,
        dynamo=dynamo,
    )


@click.command(context_settings={"show_default": True})
@click.option("--image-ckpt", type=click.Path(path_type=pathlib.Path, dir_okay=False), default=None)
@click.option("--text-ckpt", type=click.Path(path_type=pathlib.Path, dir_okay=False), default=None)
@click.option("--out-dir", type=click.Path(path_type=pathlib.Path, file_okay=False), default="models/onnx")
@click.option("--image-h", type=int, default=420)
@click.option("--image-w", type=int, default=420)
@click.option("--text-max-len", type=int, default=64)
@click.option("--opset", type=int, default=18)
@click.option("--dynamo", is_flag=True, help="Use dynamo ONNX exporter (requires onnxscript).")
def main(
    image_ckpt: pathlib.Path | None,
    text_ckpt: pathlib.Path | None,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    opset: int,
    dynamo: bool,
) -> None:
    paths = export_onnx_models(
        image_ckpt=image_ckpt,
        text_ckpt=text_ckpt,
        out_dir=out_dir,
        image_h=image_h,
        image_w=image_w,
        text_max_len=text_max_len,
        opset=opset,
        dynamo=dynamo,
    )

    click.echo("Export to ONNX is done:")
    click.echo(f"\tImage: {paths.image_onnx}")
    click.echo(f"\tText:  {paths.text_onnx}")


if __name__ == "__main__":
    main()

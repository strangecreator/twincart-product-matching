from __future__ import annotations

import pathlib

# click & related imports
import click

# utility imports
import pandas as pd

# hydra imports
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# local imports
from twincart.common.paths import ProjectPaths
from twincart.common.seed import seed_everything
from twincart.data.folds import FoldConfig, make_folds
from twincart.data.dvc_ops import ensure_exists_or_pull


def _config_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "configs"


def _load_cfg(config_name: str, overrides: list[str]) -> object:
    config_dir = _config_dir()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)

    from hydra.core.hydra_config import HydraConfig

    HydraConfig.instance().set_config(config)
    return config


def _dvc_remote_from_cfg(config: object) -> str | None:
    return str(config.data.dvc_remote) if "dvc_remote" in config.data else None


def _write_folds(config: object) -> pathlib.Path:
    paths = ProjectPaths.from_cfg(config)
    paths.ensure_dirs()

    train_csv = pathlib.Path(config.data.train_csv)
    remote = _dvc_remote_from_cfg(config)
    ensure_exists_or_pull([train_csv], remote=remote)

    frame = pd.read_csv(train_csv)
    fold_config = FoldConfig(num_folds=int(config.data.num_folds), seed=int(config.data.seed))
    folds_df = make_folds(frame, fold_config)

    out_path = pathlib.Path(config.data.folds_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_parquet(out_path, index=False)

    return out_path


@click.group(context_settings={"show_default": True})
def main() -> None:
    """TwinCart CLI (train / export / serve)."""


@main.command("prepare-folds")
@click.option("--overrides", multiple=True, help="Hydra overrides, for example, data.num_folds=5")
def prepare_folds_cmd(overrides: tuple[str, ...]) -> None:
    config = _load_cfg("train", list(overrides))
    out_path = _write_folds(config)
    click.echo(f"[Done] Wrote folds: `{out_path}`.")


@main.command("train")
@click.option("--mode", type=click.Choice(["image", "text"], case_sensitive=False), required=True)
@click.option("--fold", type=int, default=0)
@click.option("--resume", type=str, default="")
@click.option("--overrides", multiple=True)
def train_cmd(mode: str, fold: int, resume: str, overrides: tuple[str, ...]) -> None:
    config = _load_cfg("train", list(overrides) + [f"mode={mode}"])

    paths = ProjectPaths.from_cfg(config)
    paths.ensure_dirs()
    seed_everything(int(config.seed))

    remote = _dvc_remote_from_cfg(config)
    train_csv = pathlib.Path(config.data.train_csv)
    images_dir = pathlib.Path(config.data.train_images_dir)

    ensure_exists_or_pull([train_csv], remote=remote)
    if mode == "image":
        ensure_exists_or_pull([images_dir], remote=remote)

    folds_path = pathlib.Path(config.data.folds_path)
    if not folds_path.exists():
        out_path = _write_folds(config)
        click.echo(f"[Info] folds missing -> created `{out_path}`.")

    df = pd.read_parquet(folds_path)

    # additional lazy imports
    import pytorch_lightning as pl
    from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

    from twincart.train.callbacks import MlflowMetaCallback, PlotsFromCSVCallback
    from twincart.train.datamodule import TwinCartDataModule
    from twincart.train.lightning_module import TrainCfg, TwinCartModule

    class UnsafeTorchCheckpointIO(TorchCheckpointIO):
        def load_checkpoint(self, path, map_location=None, weights_only=None):
            return super().load_checkpoint(path, map_location=map_location, weights_only=False)

    mlf_logger = MLFlowLogger(
        tracking_uri=str(config.logging.mlflow.tracking_uri),
        experiment_name=str(config.logging.mlflow.experiment_name),
        run_name=f"{mode}-fold{fold}",
    )
    csv_logger = CSVLogger(save_dir=str(paths.plots_dir / "raw_logs"), name=mode)

    if mode == "image":
        batch_size = int(config.image_train.batch_size)
        lr_start = float(config.image_train.lr_start)
        lr_end = float(config.image_train.lr_end)
        use_pair_sampler = bool(config.image_train.pair_sampling.enabled)
        text_aug_p = 0.0
    else:
        batch_size = int(config.text_train.batch_size)
        lr_start = float(config.text_train.lr_start)
        lr_end = float(config.text_train.lr_end)
        use_pair_sampler = True
        text_aug_p = float(config.transforms.text.aug.p)

    datamodule = TwinCartDataModule(
        mode=mode,
        df=df,
        fold_index=fold,
        images_dir=pathlib.Path(config.data.train_images_dir),
        transforms_config=config.transforms,
        image_model_config=config.image_model,
        text_model_config=config.text_model,
        batch_size=batch_size,
        num_workers=int(config.dataloader.num_workers),
        seed=int(config.seed),
        use_pair_sampler=use_pair_sampler,
        text_aug_p=text_aug_p,
    )

    module = TwinCartModule(
        mode=mode,
        image_model_config=config.image_model,
        text_model_config=config.text_model,
        loss_config=config.loss,
        xbm_config=config.xbm,
        optim_config=config.optim,
        sched_config=config.sched,
        train_config=TrainCfg(
            mode=mode,
            lr_start=lr_start,
            lr_end=lr_end,
            max_epochs=int(config.trainer.max_epochs),
        ),
    )

    ckpt_dir = paths.models_dir / "checkpoints" / mode / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_recall1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    plot_dir = paths.plots_dir / "train" / mode / f"fold_{fold}"
    config_no_hydra = OmegaConf.masked_copy(config, [key for key in config.keys() if key != "hydra"])
    callbacks = [
        checkpoint_cb,
        MlflowMetaCallback(config=OmegaConf.to_container(config_no_hydra, resolve=True)),
        PlotsFromCSVCallback(out_dir=plot_dir),
    ]

    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=str(config.trainer.precision),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        plugins=[UnsafeTorchCheckpointIO()],
        callbacks=callbacks,
        logger=[mlf_logger, csv_logger],
    )

    ckpt_path = resume or None
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    click.echo(f"[Done] Best checkpoint: `{checkpoint_cb.best_model_path}`.")


@main.command("export-onnx", context_settings={"show_default": True})
@click.option("--image-ckpt", type=click.Path(path_type=pathlib.Path, dir_okay=False), default=None)
@click.option("--text-ckpt", type=click.Path(path_type=pathlib.Path, dir_okay=False), default=None)
@click.option("--out-dir", type=click.Path(path_type=pathlib.Path, file_okay=False), default="models/onnx")
@click.option("--image-h", type=int, default=420)
@click.option("--image-w", type=int, default=420)
@click.option("--text-max-len", type=int, default=64)
@click.option("--opset", type=int, default=18)
@click.option("--dynamo", is_flag=True, help="Use dynamo ONNX exporter (requires onnxscript).")
def export_onnx_cmd(
    image_ckpt: pathlib.Path | None,
    text_ckpt: pathlib.Path | None,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    opset: int,
    dynamo: bool,
) -> None:
    from twincart.export.cli import export_onnx_models

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


@main.command("build-mlflow", context_settings={"show_default": True})
@click.option("--image-onnx", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option("--text-onnx", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option("--tokenizer", "tokenizer_name", type=str, required=True, help="HF repo id or local path.")
@click.option("--out-dir", type=click.Path(path_type=pathlib.Path, file_okay=False), required=True)
@click.option("--image-h", type=int, default=420)
@click.option("--image-w", type=int, default=420)
@click.option("--text-max-len", type=int, default=64)
@click.option("--force", is_flag=True, help="Overwrite out-dir if exists.")
def build_mlflow_cmd(
    image_onnx: pathlib.Path,
    text_onnx: pathlib.Path,
    tokenizer_name: str,
    out_dir: pathlib.Path,
    image_h: int,
    image_w: int,
    text_max_len: int,
    force: bool,
) -> None:
    from twincart.serving.build_mlflow import build_mlflow_model

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

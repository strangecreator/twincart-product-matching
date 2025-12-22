from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from twincart.common.paths import ProjectPaths
from twincart.common.seed import seed_everything
from twincart.data.dvc_ops import ensure_exists_or_pull
from twincart.data.folds import FoldConfig, make_folds


def _config_dir() -> Path:
    # src/twincart/commands.py -> project_root/configs
    return Path(__file__).resolve().parents[2] / "configs"


def _load_cfg(config_name: str, overrides: list[str]) -> object:
    cfg_dir = _config_dir()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)

    from hydra.core.hydra_config import HydraConfig

    HydraConfig.instance().set_config(cfg)
    return cfg


def prepare_folds(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("twincart-prepare-folds")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = _load_cfg("train", overrides=list(args.overrides))
    paths = ProjectPaths.from_cfg(cfg)
    paths.ensure_dirs()

    train_csv = Path(cfg.data.train_csv)
    remote = str(cfg.data.dvc_remote) if "dvc_remote" in cfg.data else None

    ensure_exists_or_pull([train_csv], remote=remote)

    df = pd.read_csv(train_csv)
    fc = FoldConfig(num_folds=int(cfg.data.num_folds), seed=int(cfg.data.seed))
    folds_df = make_folds(df, fc)

    out_path = Path(cfg.data.folds_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_parquet(out_path, index=False)

    print(f"[OK] Wrote folds: {out_path}")


def train(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("twincart-train")
    parser.add_argument("--mode", choices=["image", "text"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    overrides = list(args.overrides)
    overrides.append(f"mode={args.mode}")

    cfg = _load_cfg("train", overrides=overrides)
    paths = ProjectPaths.from_cfg(cfg)
    paths.ensure_dirs()
    seed_everything(int(cfg.seed))

    # Ensure required data via DVC
    remote = str(cfg.data.dvc_remote) if "dvc_remote" in cfg.data else None
    train_csv = Path(cfg.data.train_csv)
    images_dir = Path(cfg.data.train_images_dir)

    ensure_exists_or_pull([train_csv], remote=remote)
    if args.mode == "image":
        ensure_exists_or_pull([images_dir], remote=remote)

    # Ensure folds exist
    folds_path = Path(cfg.data.folds_path)
    if not folds_path.exists():
        prepare_folds([])

    df = pd.read_parquet(folds_path)

    # Lazily import heavy deps only for training
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    from twincart.train.datamodule import TwinCartDataModule
    from twincart.train.lightning_module import TrainCfg, TwinCartModule

    # Per-mode params (from train.yaml)
    if args.mode == "image":
        bs = int(cfg.image_train.batch_size)
        lr_start = float(cfg.image_train.lr_start)
        lr_end = float(cfg.image_train.lr_end)
        use_pair_sampler = bool(cfg.image_train.pair_sampling.enabled)
        text_aug_p = 0.0
    else:
        bs = int(cfg.text_train.batch_size)
        lr_start = float(cfg.text_train.lr_start)
        lr_end = float(cfg.text_train.lr_end)
        use_pair_sampler = bool(cfg.text_train.batch_size > 0)  # still ok to use sampler
        text_aug_p = float(cfg.transforms.text.aug.p)

    dm = TwinCartDataModule(
        mode=args.mode,
        df=df,
        fold_index=args.fold,
        images_dir=Path(cfg.data.train_images_dir),
        transforms_cfg=cfg.transforms,
        image_model_cfg=cfg.image_model,
        text_model_cfg=cfg.text_model,
        batch_size=bs,
        num_workers=int(cfg.dataloader.num_workers),
        seed=int(cfg.seed),
        use_pair_sampler=use_pair_sampler if args.mode == "image" else True,
        text_aug_p=text_aug_p,
    )

    module = TwinCartModule(
        mode=args.mode,
        image_model_cfg=cfg.image_model,
        text_model_cfg=cfg.text_model,
        loss_cfg=cfg.loss,
        xbm_cfg=cfg.xbm,
        optim_cfg=cfg.optim,
        sched_cfg=cfg.sched,
        train_cfg=TrainCfg(
            mode=args.mode,
            lr_start=lr_start,
            lr_end=lr_end,
            max_epochs=int(cfg.trainer.max_epochs),
        ),
    )

    ckpt_dir = paths.models_dir / "checkpoints" / args.mode / f"fold_{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    (ckpt_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_recall1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        max_epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        callbacks=[ckpt],
    )

    ckpt_path = args.resume if args.resume else None
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)
    print(f"[OK] Best checkpoint: {ckpt.best_model_path}")


def _main() -> None:
    parser = argparse.ArgumentParser("twincart")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare_folds")
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=["image", "text"], required=True)
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--resume", type=str, default="")

    args, unknown = parser.parse_known_args()

    if args.cmd == "prepare_folds":
        prepare_folds(unknown)
    elif args.cmd == "train":
        train(["--mode", args.mode, "--fold", str(args.fold)] + unknown)


if __name__ == "__main__":
    _main()

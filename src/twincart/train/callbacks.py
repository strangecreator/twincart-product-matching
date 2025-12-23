from __future__ import annotations

import json
import pathlib
import typing as tp

# torch & related imports
import pytorch_lightning as pl

# utility imports
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from twincart.common.git import git_commit_id


class MlflowMetaCallback(pl.Callback):
    def __init__(self, config: tp.Any) -> None:
        self.config = config

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        commit = git_commit_id()

        # logging params and tags
        for logger in trainer.loggers:
            if logger.__class__.__name__ == "MLFlowLogger":
                logger.experiment.set_tag(logger.run_id, "git_commit", commit)

                # dumps full hydra config as artifact
                config_path = pathlib.Path(trainer.default_root_dir) / "resolved_config.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(self.config, indent=2, default=str))

                logger.experiment.log_artifact(logger.run_id, str(config_path))


class PlotsFromCSVCallback(pl.Callback):
    def __init__(self, out_dir: pathlib.Path) -> None:
        self.out_dir = out_dir

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # finding CSVLogger
        csv_logger = None
        for logger in trainer.loggers:
            if logger.__class__.__name__ == "CSVLogger":
                csv_logger = logger
                break

        if csv_logger is None:
            return

        metrics_path = pathlib.Path(csv_logger.log_dir) / "metrics.csv"
        if not metrics_path.exists():
            return

        df = pd.read_csv(metrics_path)

        # train loss (step)
        if "train_loss_step" in df.columns:
            d = df.dropna(subset=["train_loss_step"])
            if len(d):
                plt.figure()
                plt.plot(d["step"], d["train_loss_step"])
                plt.xlabel("step")
                plt.ylabel("train_loss_step")
                plt.title("Train loss (step)")
                plt.tight_layout()
                plt.savefig(self.out_dir / "train_loss_step.png")
                plt.close()

        # val loss (epoch)
        if "val_loss" in df.columns:
            d = df.dropna(subset=["val_loss"])
            if len(d):
                plt.figure()
                plt.plot(d["epoch"], d["val_loss"])
                plt.xlabel("epoch")
                plt.ylabel("val_loss")
                plt.title("Val loss (epoch)")
                plt.tight_layout()
                plt.savefig(self.out_dir / "val_loss.png")
                plt.close()

        # val metric (epoch)
        if "val_recall1" in df.columns:
            d = df.dropna(subset=["val_recall1"])
            if len(d):
                plt.figure()
                plt.plot(d["epoch"], d["val_recall1"])
                plt.xlabel("epoch")
                plt.ylabel("val_recall1")
                plt.title("Val Recall@1 (epoch)")
                plt.tight_layout()
                plt.savefig(self.out_dir / "val_recall1.png")
                plt.close()

        # learning rate curve
        if "lr" in df.columns:
            d = df.dropna(subset=["lr"])
            if len(d):
                plt.figure()
                plt.plot(d["step"], d["lr"])
                plt.xlabel("step")
                plt.ylabel("lr")
                plt.title("Learning rate")
                plt.tight_layout()
                plt.savefig(self.out_dir / "lr.png")
                plt.close()

import json
import os
from pathlib import Path
from typing import Mapping

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from gjepa.config import TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_trainer(
    config: TrainingConfig,
    reload_dataloaders_every_n_epochs: int = 0,
) -> Trainer:
    loggers = setup_loggers(config)
    callbacks = setup_callbacks(config)
    return Trainer(
        default_root_dir=config.experiment_dir,
        log_every_n_steps=1,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        deterministic="warn",
        accelerator=DEVICE,
        devices=1,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )


def setup_callbacks(config: TrainingConfig) -> list[Callback]:
    callbacks: list[Callback] = [LearningRateMonitor(logging_interval="epoch")]
    if config.early_stopping:
        callbacks.append(EarlyStopping(**config.early_stopping))
    if config.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            filename=f"{{epoch}}-{{{config.checkpoint['monitor']}:.2f}}",
            **config.checkpoint,
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)

    callbacks.append(ModelSummary(max_depth=3))

    return callbacks


def setup_loggers(config: TrainingConfig) -> list[Logger]:
    # need use random string, since wandb won't accept same id in the future (also after delete)
    version = os.getenv("EXP_NAME", f"version_{config.random_seed}")

    loggers: list[Logger] = []

    if config.use_tensorboard:
        tb_logger = TensorBoardLogger(
            str(config.experiment_dir), default_hp_metric=False, version=version
        )
        loggers.append(tb_logger)

    if config.use_wandb:
        assert config.wandb_project is not None
        config.experiment_dir.mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbLogger(
            name=config.experiment_name,
            save_dir=config.experiment_dir,
            project=config.wandb_project,
            log_model=True,
            reinit=True,
        )
        loggers.append(wandb_logger)
    return loggers


def save_metrics(metrics: Mapping[str, float], log_dir: Path) -> None:
    metric_path = log_dir / "metrics.json"
    with metric_path.open("w") as file:
        json.dump(metrics, file, indent="\t")

from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar, get_args

import pytorch_lightning as pl
from pydantic import BaseModel
from torch import Tensor
from torch_geometric.data import Data

from gjepa.config import DatasetConfig, TrainingConfig, GraphLevelDatasetConfig, PosEncodingConfig
from gjepa.models.downstream import LinearProbingClassifier, LinearProbingRegressor

Cfg = TypeVar("Cfg", bound="SSLConfigBase")
Out = TypeVar("Out")


class SSLConfigBase(BaseModel, extra="forbid"):
    dataset: DatasetConfig
    training: TrainingConfig
    backbone: dict[str, Any]
    check_val_every_n_epoch: int = 10

    @property
    def version(self) -> str:
        return f"version_{self.training.random_seed}"

class SSLGraphLevelConfigBase(BaseModel, extra="forbid"):
    dataset: GraphLevelDatasetConfig
    training: TrainingConfig
    pos_encoding: PosEncodingConfig | None = None
    backbone: dict[str, Any]
    check_val_every_n_epoch: int = 10

    @property
    def version(self) -> str:
        return f"version_{self.training.random_seed}"

class SSLModelBase(pl.LightningModule, ABC, Generic[Cfg, Out]):
    """Base class for SSL models."""

    type_Cfg: Type[Cfg]

    def __init__(self, config: Cfg | dict[str, Any]) -> None:
        super(SSLModelBase, self).__init__()

        if isinstance(config, dict):
            config = self.type_Cfg(**config)
        self.save_hyperparameters(config.model_dump())

        self.config = config

        self.downstream_model = LinearProbingRegressor(
            self.config.dataset.task_type,
            self.config.dataset.out_channels,
        )

    def __init_subclass__(cls) -> None:
        cls.type_Cfg = get_args(cls.__orig_bases__[0])[0]  # type: ignore

    @abstractmethod
    def forward(self, x: Tensor, edge_index: Tensor) -> Out:
        return NotImplemented

    @abstractmethod
    def forward_repr(self, x: Tensor, edge_index: Tensor, batch) -> Tensor:
        return NotImplemented

    def on_validation_epoch_start(self) -> None:
        self._update_downstream_model_with_train_representations()

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        z, y = self._get_z_y_from_batch(batch)
        self.downstream_model.update_test(z, y)

    def on_validation_epoch_end(self) -> None:
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="val_")
        print(metrics)
        self.log_dict(metrics)

    def on_test_start(self) -> None:
        self._update_downstream_model_with_train_representations()

    def test_step(self, batch: Data, batch_idx: int) -> None:
        z, y = self._get_z_y_from_batch(batch)
        self.downstream_model.update_test(z, y)

    def on_test_epoch_end(self) -> None:
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="test_")
        print(metrics)
        self.log_dict(metrics)

    def _update_downstream_model_with_train_representations(self) -> None:
        self.downstream_model.reset()
        for batch in self.trainer.datamodule.train_inference_dataloader():  # type: ignore
            # Iterating data_loader manually requires manual device change:
            z, y = self._get_z_y_from_batch(batch.to(self.device))
            self.downstream_model.update_train(z, y)

    def _get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.forward_repr(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
        y = batch.y
        return z[: batch.batch_size], y[: batch.batch_size]

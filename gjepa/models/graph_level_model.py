from typing import Any

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from gjepa.config import GraphLevelExperimentConfig
from gjepa.models.encoders import GNNEncoder
from gjepa.models.predictors import LinearClassifier, LinearRegressor
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore


class SupervisedGraphLevelGNN(LightningModule):
    """Graph-level training."""

    def __init__(self, config: GraphLevelExperimentConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = GraphLevelExperimentConfig.from_raw_config(config)

        self.save_hyperparameters({"config": config.model_dump()})
        self.config = config

        self.gnn = GNNEncoder(**self.config.model.backbone)

        task_type = self.config.dataset.task_type

        if task_type.endswith("regression"):
            predictor_cls = LinearRegressor
        else:
            predictor_cls = LinearClassifier
        self.predictor = predictor_cls(
            in_channels=self.gnn.out_channels,
            out_channels=self.config.dataset.out_channels,
            task_type=task_type,
        )

        self.metrics = nn.ModuleDict(
            {
                f"{split}_metrics": self.predictor.metrics.clone(prefix=f"{split}_")
                for split in ("train", "val", "test")
            }
        )

    def forward(self, batch: Data) -> Tensor:
        z = self._get_pooled_z(batch)

        logits = self.predictor(z)

        return logits

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, split="train")
        self.log("train_loss", loss, batch_size=batch.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self._shared_step(batch, split="val")

    def test_step(self, batch: Data, batch_idx: int) -> None:
        self._shared_step(batch, split="test")

    def predict_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, Tensor]:
        return self.get_z_y_from_batch(batch)

    def _shared_step(self, batch: Data, split: str) -> Tensor:
        logits = self.forward(batch)

        y_gt = batch.y
        loss = self.predictor.loss_func(input=logits, target=y_gt)
        self.metrics[f"{split}_metrics"](preds=logits, target=y_gt)
        self.log_dict(
            self.metrics[f"{split}_metrics"],  # type: ignore[arg-type]
            batch_size=len(y_gt),
        )

        return loss

    def _get_pooled_z(self, batch: Data) -> Tensor:
        z = self.gnn(
            batch=batch
            
        )

        z = global_mean_pool(z, batch.batch)

        return z

    def get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self._get_pooled_z(batch)
        y = batch.y

        return z, y

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        optim = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        if self.config.training.scheduler_config is None:
            return {"optimizer": optim}

        scheduler = LinearWarmupCosineAnnealingLR(optim, **self.config.training.scheduler_config)
        return {"optimizer": optim, "lr_scheduler": scheduler}


    def setup(self, stage: str):
        dm = getattr(self.trainer, "datamodule", None)
        y_std = getattr(dm, "y_std", None) if dm is not None else None
        if y_std is None:
            return
        import torch
        y_std_t = torch.as_tensor(y_std, dtype=torch.float32)

        from gjepa.metrics.regression import get_default_regression_metrics
        self.predictor.metrics = get_default_regression_metrics(
            task_type=self.config.dataset.task_type,
            output_dim=self.config.dataset.out_channels,
            y_std=y_std_t,
            reduce_mean=True if self.config.dataset.task_type == "multiregression" else False,
        )
        self.metrics = nn.ModuleDict({
            f"{split}_metrics": self.predictor.metrics.clone(prefix=f"{split}_")
            for split in ("train", "val", "test")
        })

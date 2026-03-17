from typing import Any

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch_geometric.data import Data

from gjepa.config import ExperimentConfig
from gjepa.models.encoders import GNNEncoder
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore


class SupervisedNodeLevelGNN(LightningModule):
    """Node-level training."""

    def __init__(self, config: ExperimentConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = ExperimentConfig.from_raw_config(config)

        self.save_hyperparameters({"config": config.model_dump()})
        self.config = config

        self.gnn = GNNEncoder(**self.config.model.backbone)
        self.predictor = LinearClassifier(
            in_channels=self.gnn.out_channels,
            out_channels=self.config.dataset.out_channels,
            task_type=self.config.dataset.task_type,
        )

        self.metrics = nn.ModuleDict(
            {
                f"{split}_metrics": self.predictor.metrics.clone(prefix=f"{split}_")
                for split in ("train", "val", "test")
            }
        )

    def forward(self, batch: Data) -> Tensor:
        z = self.gnn(x=batch.x, edge_index=batch.edge_index, pos=batch.pos, batch=batch.batch)
        z = z[: batch.batch_size]

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
        z = self.gnn(x=batch.x, edge_index=batch.edge_index, pos=batch.pos, batch=batch.batch)
        return z[: batch.batch_size], batch.y[: batch.batch_size]

    def _shared_step(self, batch: Data, split: str) -> Tensor:
        logits = self.forward(batch)

        y_gt = batch.y[: batch.batch_size]
        loss = self.predictor.loss_func(input=logits, target=y_gt)

        self.metrics[f"{split}_metrics"](preds=logits, target=y_gt)
        self.log_dict(
            self.metrics[f"{split}_metrics"],  # type: ignore[arg-type]
            batch_size=len(y_gt),
        )

        return loss

    def get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.gnn(x=batch.x, edge_index=batch.edge_index)
        y = batch.y
        return z[: batch.batch_size], y[: batch.batch_size]

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

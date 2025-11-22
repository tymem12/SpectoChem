from typing import Any, Literal, Optional

import os
import pandas as pd 
import numpy as np
import torch


from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from gjepa.config import GraphLevelExperimentConfig
from gjepa.models.encoders import GNNEncoder
from gjepa.models.predictors import LinearClassifier, LinearRegressor
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore
from gjepa.utils.cv_vis import plot_graph_with_predictions

class SupervisedGraphLevelGNN(LightningModule):
    """Graph-level training."""

    def __init__(self, config: GraphLevelExperimentConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = GraphLevelExperimentConfig.from_raw_config(config)

        self.save_hyperparameters({"config": config.model_dump()})
        self.config = config

        self.gnn = GNNEncoder(**self.config.model.backbone)

        ds_config = self.config.dataset

        task_type = ds_config.task_type

        if task_type.endswith("regression"):
            predictor_cls = LinearRegressor
            predictor_kwargs = dict(
                prediction_type=self._get_prediction_type(),
                #spectral_loss="sid",
                #threshold=1e-8
            )
        else:
            predictor_cls = LinearClassifier
            predictor_kwargs = {}

        self.predictor = predictor_cls(
            in_channels=self.gnn.out_channels,
            out_channels=ds_config.out_channels,
            task_type=task_type,
            **predictor_kwargs
        )

        self.metrics = nn.ModuleDict(
            {
                f"{split}_metrics": self.predictor.metrics.clone(prefix=f"{split}_")
                for split in ("train", "val", "test")
            }
        )
        self._test_outputs: list[dict] = []

    def _get_prediction_type(
        self
    ) -> Optional[Literal["pairs", "vector"]]:
        return self.config.dataset.additional_loading_params.get(
            "prediction_type", None
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

    def test_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, split="test")
        logits = self.forward(batch)
        self._test_outputs.append(
            {
                "origin_id": batch.origin_id,
                "y": batch.y.detach().cpu(),
                "y_pred": logits.detach().cpu(),
            }
        )

        return loss

    def predict_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, Tensor]:
        return self.get_z_y_from_batch(batch)

    def _shared_step(self, batch: Data, split: str) -> Tensor:
        logits = self.forward(batch)
        y_gt = batch.y
        loss = self.predictor.loss_func(input=logits, target=y_gt)

        metrics = self.metrics[f"{split}_metrics"]
        metrics(preds=logits, target=y_gt)

        # --- Log metrics (support non-reduced outputs) ---
        computed_metrics = metrics.compute()
        log_dict = {}

        for name, value in computed_metrics.items():
            # If the metric returns a tensor with >0 dims (e.g., per-target metrics)
            if torch.is_tensor(value) and value.ndim > 0:
                for i, v in enumerate(value):
                    log_dict[f"{name}_{i}"] = v
            else:
                log_dict[name] = value

        # Log to TensorBoard
        self.log_dict(log_dict, batch_size=len(y_gt))

        return loss

    def on_test_end(self) -> None:

        if len(self._test_outputs) == 0:
            return

        all_ids: list[str] = []
        for o in self._test_outputs:
            all_ids.extend(list(o["origin_id"]))
        y_all = torch.cat([o["y"] for o in self._test_outputs], dim=0)
        y_pred_all = torch.cat([o["y_pred"] for o in self._test_outputs], dim=0)
        N, D = y_all.shape
        assert len(all_ids) == N, f'N: {N}, len(all_ids): {len(all_ids)}'

        data = {"origin_id": np.array(all_ids)}
        y_np = y_all.numpy()
        y_pred_np = y_pred_all.numpy()

        for i in range(D):
            data[f"target_{i}"] = y_np[:, i]
            data[f"prediction_{i}"] = y_pred_np[:, i]

        df = pd.DataFrame(data)

        save_dir = (
            self.trainer.logger.log_dir
            if self.trainer is not None and self.trainer.logger is not None
            else self.trainer.default_root_dir
        )
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "test_predictions_wide.csv")
        df.to_csv(csv_path, index=False)
        output_params = self.config.dataset.additional_loading_params

        plot_graph_with_predictions(df, output_params['prediction_type'],save_dir, tuple(output_params['vis_range']))
        print(f"[SupervisedGraphLevelGNN] saved predictions to: {csv_path}")


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

        ds_config = self.config.dataset

        from gjepa.metrics.regression import get_default_regression_metrics
        self.predictor.metrics = get_default_regression_metrics(
            task_type=ds_config.task_type,
            output_dim=ds_config.out_channels,
            y_std=y_std_t,
            prediction_type=self._get_prediction_type(),
            reduce_mean=ds_config.task_type == "multiregression"
        )
        self.metrics = nn.ModuleDict({
            f"{split}_metrics": self.predictor.metrics.clone(prefix=f"{split}_")
            for split in ("train", "val", "test")
        })

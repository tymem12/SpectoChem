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
from gjepa.models.predictors import MLPClassifier, MLPRegressor
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore
from gjepa.utils.cv_vis import plot_graph_with_predictions
from gjepa.datasets.cv_vis.standarizer_singleton import StandarizerSingletonF, StandarizerSingletonLambda


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
        predictor_kwargs = ds_config.predictor_kwargs

        if predictor_kwargs is None:
            predictor_kwargs = {}
        else:
            predictor_kwargs = predictor_kwargs.copy()

        task_type = ds_config.task_type

        if task_type.endswith("regression"):
            predictor_cls = MLPRegressor
            predictor_kwargs |= dict(
                prediction_type=self._get_prediction_type(),
                #spectral_loss="sid",
                #threshold=1e-8
            )
        else:
            predictor_cls = MLPClassifier

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
        self._val_outputs: list[dict] = []   # <--- added

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
        loss = self._shared_step(batch, split="val")
        self.log("val_loss", loss, batch_size=batch.batch_size, prog_bar=False)


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
        self.log("test_loss", loss, batch_size=batch.batch_size, prog_bar=False)
        standarized_lambda_val = StandarizerSingletonLambda.get_values()
        standarized_f_val = StandarizerSingletonF.get_values()
        if standarized_lambda_val['standarize']:
            self.log('lambda_mean',standarized_lambda_val['mean_lambda'])
            self.log('lambda_std',standarized_lambda_val['std_lambda'])
        if standarized_f_val['standarize']:
            self.log('f_mean',standarized_f_val['mean_f'])
            self.log('f_std',standarized_f_val['std_f'])


        return loss

    def predict_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, Tensor]:
        return self.get_z_y_from_batch(batch)

    def _shared_step(self, batch: Data, split: str) -> Tensor:
        logits = self.forward(batch)
        y_gt = batch.y  # whatever the dataset gave us

        task_type = self.config.dataset.task_type

        if task_type in ("multilabel", "binary", "binary_multitask", 'multiregression', 'regression'):
            y_loss = y_gt.float()
        elif task_type == "multiclass":
            y_loss = y_gt.long()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        loss = self.predictor.loss_func(input=logits, target=y_loss)

    
        probas = self.predictor.logits_to_proba(logits)
        metrics = self.metrics[f"{split}_metrics"]
        metrics.update(preds=probas, target=y_gt)

        self.log_dict(
            metrics,
            on_step=(split == "train"),  
            on_epoch=True,               
            batch_size=len(y_gt),
        )

        return loss


    def on_test_end(self) -> None:

        output_params = self.config.dataset.additional_loading_params

        if output_params['prediction_type'] == 'binary_vector_multiclass':
            return
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

        plot_graph_with_predictions(df, output_params['prediction_type'],save_dir, tuple(output_params['vis_range']))
        print(f"[SupervisedGraphLevelGNN] saved predictions to: {csv_path}")


    def _get_pooled_z(self, batch: Data) -> Tensor:
        z = self.gnn(batch=batch)

        # UMA returned representation on the GRAPH LEVEL [num_graphs, emebding] when schent returns representation
        # on the ATOM LEVEL [n.atoms, embeding] with are global_mean_pooled into [num_graphs, emebding]
        if z.dim() == 2 and z.size(0) == batch.num_graphs:
            return z
        if z.dim() == 2 and z.size(0) == batch.num_nodes:
            return global_mean_pool(z, batch.batch)
        n_graphs = int(batch.batch.max().item()) + 1
        if z.dim() == 2 and z.size(0) == n_graphs:
            return z

        raise RuntimeError(
            f"Can't pool: z.shape={tuple(z.shape)}, "
            f"num_nodes={batch.num_nodes}, num_graphs={batch.num_graphs}, "
            f"batch.batch.shape={tuple(batch.batch.shape)}"
        )

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

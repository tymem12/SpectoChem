import json
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pytorch_lightning as pl
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec

from gjepa.config import DatasetConfig
from gjepa.datasets.node_level import load_graph
from gjepa.utils import resolve_config
from gjepa.utils.pyg import create_transform


class Node2vecModel(pl.LightningModule):
    """Node2vec wrapper for Pytorch-Lightning."""

    def __init__(
        self,
        graph: Data,
        embedding_dim: int,
        n2v_hparams: dict[str, float | int],
    ):
        super().__init__()

        self.graph = graph
        self.model = Node2Vec(
            edge_index=graph.edge_index,
            embedding_dim=embedding_dim,
            **n2v_hparams,
        )

    def forward_repr(self) -> torch.Tensor:
        return self.model.embedding.weight

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        pos_rw, neg_rw = batch
        loss = self.model.loss(pos_rw=pos_rw, neg_rw=neg_rw)

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            z = self.forward_repr().cpu()

        y = self.graph.y
        train_mask = self.graph.train_mask
        val_mask = self.graph.val_mask

        lr = LogisticRegression()
        lr.fit(z[train_mask], y[train_mask])

        y_score_val = lr.predict_proba(z[val_mask])
        y_pred_val = y_score_val.argmax(axis=1)

        if y_score_val.shape[1] == 2:
            y_score_val = y_score_val[:, 1]

        y_true_val = y[val_mask]

        metrics = {
            "val/Accuracy": accuracy_score(y_true=y_true_val, y_pred=y_pred_val),
            "val/F1": f1_score(y_true=y_true_val, y_pred=y_pred_val, average="macro"),
            "val/AUROC": roc_auc_score(
                y_true=y_true_val, y_score=y_score_val, average="macro", multi_class="ovo"
            ),
        }

        self.log_dict(metrics)

    def predict_step(
        self,
        batch: Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        with torch.no_grad():
            return self.forward_repr().cpu()

    def train_dataloader(self) -> DataLoader:
        return self.model.loader(batch_size=128)

    def val_dataloader(self) -> DataLoader:
        return self._dummy_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self._dummy_dataloader()

    def _dummy_dataloader(self) -> DataLoader:
        return DataLoader(torch.arange(1))  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-2, weight_decay=5e-4)


class Config(BaseModel, extra="forbid"):
    dataset: DatasetConfig
    random_seed: int
    embedding_dim: int
    max_epochs: int
    hyperparameters: dict[str, float | int]
    embedding_file: Path
    metrics_file: Path


@hydra.main(version_base="1.3", config_path="../../config", config_name="node2vec_embedding")
def main(cfg: DictConfig) -> None:
    config = Config(**resolve_config(cfg))

    seed_everything(config.random_seed)
    config.embedding_file.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    graph = load_graph(
        root_dir=config.dataset.root_dir,
        name=config.dataset.name,
        transform=create_transform(config.dataset.transforms),
        pre_transform=create_transform(config.dataset.pre_transforms),
    )

    model = Node2vecModel(
        graph=graph,
        embedding_dim=config.embedding_dim,
        n2v_hparams=config.hyperparameters,
    )

    with TemporaryDirectory() as tmpdir:
        trainer = pl.Trainer(
            logger=False,
            max_epochs=config.max_epochs,
            callbacks=[
                ModelCheckpoint(
                    dirpath=tmpdir,
                    filename="model",
                    monitor=f"val/{config.dataset.main_metric}",
                    mode="max",
                    verbose=True,
                ),
            ],
            deterministic="warn",
        )
        trainer.fit(model=model)

        assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
        assert isinstance(trainer.checkpoint_callback.best_model_score, Tensor)
        monitor_metric = f"val/{config.dataset.main_metric}"
        monitor_metric_val = trainer.checkpoint_callback.best_model_score.item()

        metrics = {
            **config.hyperparameters,
            monitor_metric: monitor_metric_val,
        }

        with config.metrics_file.open("w") as fout:
            json.dump(obj=metrics, fp=fout, indent=4)

        node_emb: Tensor = trainer.predict(model=model, ckpt_path="best")[0]  # type: ignore
        torch.save(obj=node_emb, f=config.embedding_file)


if __name__ == "__main__":
    main()

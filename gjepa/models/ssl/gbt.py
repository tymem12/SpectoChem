from typing import Any

import torch
from attr import dataclass
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dropout_edge

from gjepa.models.encoders import GNNEncoder
from gjepa.models.ssl.base import SSLConfigBase, SSLModelBase, SSLGraphLevelConfigBase
from gjepa.transforms.ssl import drop_features, drop_features_two_views, add_noise_positions, \
    add_noise_positional_encoding
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore


class BarlowTwinsConfig(SSLGraphLevelConfigBase):
    drop_edge_pa: float
    drop_feat_px: float
    std: float


@dataclass(kw_only=True)
class GBTOutput:
    z_a: Tensor
    z_b: Tensor


class GBTModel(SSLModelBase[BarlowTwinsConfig, GBTOutput]):
    """Graph Barlow Twins (GBT) (https://arxiv.org/abs/2106.02466)."""

    def __init__(self, config: BarlowTwinsConfig | dict[str, Any]) -> None:
        super().__init__(config)

        self.encoder = GNNEncoder(**self.config.backbone)
        self.loss = BarlowTwinsLoss()

    def forward(self, batch):
        batch_a = self.augment(batch)
        batch_b = self.augment(batch)
        z_a = self.encoder(batch_a)
        z_b = self.encoder(batch_b)
        return GBTOutput(z_a=z_a, z_b=z_b)

    def forward_repr(self, batch) -> Tensor:
        return self.encoder(batch)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        z_a = out.z_a
        z_b = out.z_b

        loss = self.loss(z_a, z_b)
        self.log("train/loss", loss)

        return loss

    def augment(self, batch):
        batch_aug = batch.clone()
        batch_aug.x, batch_aug.z = drop_features_two_views(x=batch.x, z=batch.z, p=self.config.drop_feat_px)

        if batch.positional_encoding is not None:
            batch_aug.edge_index, _ = dropout_edge(
                edge_index=batch.edge_index, p=self.config.drop_edge_pa
            )
            # batch_aug.positional_encoding = add_noise_positional_encoding(
            #     batch.positional_encoding, p=self.config.drop_edge_pa, rel_std=self.config.std
            # )
        else:
            batch_aug.pos = add_noise_positions(
                batch.pos, p=self.config.drop_edge_pa, std=self.config.std
            )
        return batch_aug

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            **self.config.training.scheduler_config,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss for GBT model (https://arxiv.org/abs/2106.02466).
    Credit: https://github.com/pbielak/graph-barlow-twins/blob/master/gssl/loss.py
    """

    EPS = 1e-15

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        batch_size = z_a.size(0)
        feature_dim = z_a.size(1)
        _lambda = 1 / feature_dim

        # Apply batch normalization
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + self.EPS)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + self.EPS)

        # Cross-correlation matrix
        c = (z_a_norm.T @ z_b_norm) / batch_size

        # Loss function
        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        return (1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum()


class GBTGraphModelPrePool(GBTModel):
    """Pools node embeddings before Barlow Twins loss (graph-level)."""

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        z_a = global_mean_pool(out.z_a, batch.batch)
        z_b = global_mean_pool(out.z_b, batch.batch)

        loss = self.loss(z_a, z_b)

        self.log("train/loss", loss)
        return loss

    def _get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.forward_repr(batch)
        y = batch.y
        return z, y

    def forward_repr(self, batch) -> Tensor:
        node_embeddings = self.encoder(batch)
        return global_mean_pool(node_embeddings, batch.batch)


class GBTGraphModelPostPool(GBTModel):
    """Applies Barlow Twins at node level, then pools embeddings (graph-level)."""
    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        z_a = out.z_a
        z_b = out.z_b
        loss = self.loss(z_a, z_b)
        self.log("train/loss", loss)

        return loss

    def _get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.forward_repr(batch)
        y = batch.y
        return z, y

    def forward_repr(self, batch) -> Tensor:
        node_embeddings = self.encoder(batch)
        return global_mean_pool(node_embeddings, batch.batch)



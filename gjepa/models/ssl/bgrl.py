
from typing import Any

import torch
from attr import dataclass
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, cosine_similarity
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import MLP
from torch_geometric.utils import dropout_edge

from gjepa.models.ema import EMAModel
from gjepa.models.encoders import GNNEncoder
from gjepa.models.ssl.base import SSLConfigBase, SSLModelBase, SSLGraphLevelConfigBase
from gjepa.models.ssl.gbt import GBTGraphModelPrePool
from gjepa.transforms.ssl import drop_features, drop_features_two_views, add_noise_positional_encoding, \
    add_noise_positions
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore


class BGRLConfig(SSLGraphLevelConfigBase):
    predictor: dict[str, Any]
    drop_edge_p1: float
    drop_feat_p1: float
    std_1: float
    drop_edge_p2: float
    drop_feat_p2: float
    std_2: float

@dataclass(kw_only=True)
class BGRLOutput:
    h1: Tensor
    h2: Tensor
    h1_pred: Tensor
    h2_pred: Tensor
    h1_target: Tensor
    h2_target: Tensor


class BGRLModel(SSLModelBase[BGRLConfig, BGRLOutput]):
    """Bootstrapped Graph Latents (BGRL) (https://arxiv.org/abs/2102.06514)."""

    EMA_TEMPERATURE = 0.99

    def __init__(self, config: BGRLConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self.online_encoder = GNNEncoder(**self.config.backbone)
        self.target_encoder = EMAModel(
            self.online_encoder,
            schedule="cosine",
            max_steps=self.config.training.max_epochs,
            temperature=self.EMA_TEMPERATURE,
        )
        channel_list = [
            self.online_encoder.out_channels,
            self.config.predictor.pop("hidden_channels"),
            self.online_encoder.out_channels,
        ]
        self.predictor = MLP(
            channel_list=channel_list,
            **self.config.predictor,
        )

    def forward(self, batch) -> BGRLOutput:
        batch_aug_1 = self.augment(batch, self.config.drop_feat_p1, self.config.drop_edge_p1, self.config.std_1)
        batch_aug_2 = self.augment(batch, self.config.drop_feat_p2, self.config.drop_edge_p2, self.config.std_2)

        h1 = self.online_encoder(batch_aug_1)
        h2 = self.online_encoder(batch_aug_2)

        h1_pred = self.predictor(h1)
        h2_pred = self.predictor(h2)

        with torch.no_grad():
            h1_target = self.target_encoder(batch_aug_1)
            h2_target = self.target_encoder(batch_aug_2)

        return BGRLOutput(
            h1=h1,
            h2=h2,
            h1_pred=h1_pred,
            h2_pred=h2_pred,
            h1_target=h1_target,
            h2_target=h2_target,
        )

    def forward_repr(self, batch) -> Tensor:
        return self.online_encoder(batch)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        h1_pred = out.h1_pred[: batch.batch_size]
        h2_pred = out.h2_pred[: batch.batch_size]

        h1_target = out.h1_target[: batch.batch_size]
        h2_target = out.h2_target[: batch.batch_size]

        loss = (
            2
            - cosine_similarity(h1_pred, h1_target.detach(), dim=-1).mean()
            - cosine_similarity(h2_pred, h2_target.detach(), dim=-1).mean()
        )

        self.log("train/loss", loss)

        return loss

    def on_train_epoch_end(self) -> None:
        """Updates target network with EMA."""
        self.target_encoder.update_model(self.online_encoder, self.current_epoch)
        self.log("train/EMA_momentum", self.target_encoder.last_momentum)

    def augment(self, batch, drop_feat_p, drop_edge_p, std):
        batch_aug = batch.clone()
        batch_aug.x, batch_aug.z = drop_features_two_views(x=batch.x, z=batch.z, p=drop_feat_p)

        if batch.positional_encoding is not None:
            batch_aug.edge_index, _ = dropout_edge(
                edge_index=batch.edge_index, p=drop_edge_p
            )

            # batch_aug.positional_encoding = add_noise_positional_encoding(
            #     batch.positional_encoding, p=drop_edge_p, rel_std=std
            # )
        else:
            batch_aug.pos = add_noise_positions(
                batch.pos, p=drop_edge_p, std=std
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


class BGRLGraphModelPrePool(BGRLModel):
    """
    Variant where graph-level loss is computed.
    """
    def _get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.forward_repr(batch)
        y = batch.y
        return z, y

    def forward_repr(self, batch) -> Tensor:
        h = self.online_encoder(batch)
        return global_mean_pool(h, batch.batch)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        h1_pred = global_mean_pool(out.h1_pred, batch.batch)
        h2_pred = global_mean_pool(out.h2_pred, batch.batch)

        h1_target = global_mean_pool(out.h1_target, batch.batch)
        h2_target = global_mean_pool(out.h2_target, batch.batch)

        loss = (
                2
                - cosine_similarity(h1_pred, h1_target.detach(), dim=-1).mean()
                - cosine_similarity(h2_pred, h2_target.detach(), dim=-1).mean()
        )

        self.log("train/loss", loss)
        return loss


class BGRLGraphModelPostPool(BGRLModel):
    """
    Variant where node-level encoder runs as usual,
    but forward_repr returns pooled graph representation.
    Loss is still computed node-level
    """
    def _get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.forward_repr(batch)
        y = batch.y
        return z, y

    def forward_repr(self, batch) -> Tensor:
        h = self.online_encoder(batch)
        return global_mean_pool(h, batch.batch)


    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        out = self.forward(batch)

        h1_pred = out.h1_pred[: batch.batch_size]
        h2_pred = out.h2_pred[: batch.batch_size]

        h1_target = out.h1_target[: batch.batch_size]
        h2_target = out.h2_target[: batch.batch_size]

        loss = (
            2
            - cosine_similarity(h1_pred, h1_target.detach(), dim=-1).mean()
            - cosine_similarity(h2_pred, h2_target.detach(), dim=-1).mean()
        )

        self.log("train/loss", loss)

        return loss
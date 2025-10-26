from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP
from torch_geometric.utils import dropout_edge

from gjepa.config import DatasetConfig, T_dataloader_mode
from gjepa.datasets.node_level import KHopDatamodule
from gjepa.models.encoders import GNNEncoder
from gjepa.models.ssl.base import SSLConfigBase, SSLModelBase
from gjepa.transforms.ssl import mask_nodes


class KHopDatamoduleWithStructAndPCAEmbs(KHopDatamodule):
    def __init__(
        self,
        struct_emb_path: Path,
        pca_emb_path: Path,
        dataset_config: DatasetConfig,
        batch_size: int,
        dataloader_mode: T_dataloader_mode,
        pos_enc_path: Path | None = None,
    ):
        super().__init__(dataset_config, batch_size, dataloader_mode, pos_enc_path)
        self.struct_emb_path = struct_emb_path
        self.pca_emb_path = pca_emb_path

    def _load_dataset(self) -> Data:
        data = super()._load_dataset()
        data.struct_embs = torch.load(self.struct_emb_path).data
        data.feat_pca_embs = torch.load(self.pca_emb_path)
        return data

    @property
    def struct_emb_dim(self) -> int:
        assert self.dataset is not None
        return self.dataset.struct_embs.size(1)

    @property
    def feat_pca_dim(self) -> int:
        assert self.dataset is not None
        return self.dataset.feat_pca_embs.size(1)


class GiGaMAEConfig(SSLConfigBase):
    predictor: dict[str, Any]
    struct_emb_path: Path
    feat_pca_emb_path: Path
    struct_emb_dim: int | None = None
    feat_pca_emb_dim: int | None = None
    feat_pca_ratio: float
    mask_node_prob: float
    mask_edge_prob: float
    tau: float
    l1_e: float
    l2_e: float
    l12_e: float
    l1_f: float
    l2_f: float
    l12_f: float
    l1_b: float
    l2_b: float
    l12_b: float


class GiGaMAE(SSLModelBase[GiGaMAEConfig, Tensor]):
    """Generalizable Graph Masked Autoencoder (GiGaMAE) (https://arxiv.org/abs/2308.09663)."""

    def __init__(self, config: GiGaMAEConfig | dict[str, Any]) -> None:
        super().__init__(config)
        assert self.config.struct_emb_dim is not None
        assert self.config.feat_pca_emb_dim is not None
        self.gae = GAE(
            encoder_config=self.config.backbone,
            predictor_config=self.config.predictor,
            emb_1_dim=self.config.struct_emb_dim,
            emb_2_dim=self.config.feat_pca_emb_dim,
        )
        self.loss = GiGaMAELoss(
            tau=self.config.tau,
            l1_e=self.config.l1_e,
            l2_e=self.config.l2_e,
            l12_e=self.config.l12_e,
            l1_f=self.config.l1_f,
            l2_f=self.config.l2_f,
            l12_f=self.config.l12_f,
            l1_b=self.config.l1_b,
            l2_b=self.config.l2_b,
            l12_b=self.config.l12_b,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward_repr(x=x, edge_index=edge_index)

    def forward_repr(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.gae.encode(x=x, edge_index=edge_index)

    def training_step(self, batch: Data) -> Tensor:
        x_m, ei_m, mask_both_node_edge, mask_sole_node, mask_sole_edge = self.augment(
            batch.x,
            batch.edge_index,
        )
        z = self.forward(x=x_m, edge_index=ei_m)
        z = z[: batch.batch_size]
        recon_emb_1, recon_emb_2, recon_12 = self.gae.decode_all(z)

        loss = self.loss(
            emb_1=batch.struct_embs[: batch.batch_size],
            emb_2=batch.feat_pca_embs[: batch.batch_size],
            recon_emb_1=recon_emb_1,
            recon_emb_2=recon_emb_2,
            recon_12=recon_12,
            mask_sole_node=mask_sole_node[: batch.batch_size],
            mask_sole_edge=mask_sole_edge[: batch.batch_size],
            mask_both_node_edge=mask_both_node_edge[: batch.batch_size],
        )

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def augment(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_m, mask_x = mask_nodes(x, p=self.config.mask_node_prob)
        ei_m, mask_ei = dropout_edge(edge_index, p=self.config.mask_edge_prob)

        # undirected graph assumed
        node_with_masked_edges = edge_index[0][mask_ei]
        mask_node_with_masked_edges = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        mask_node_with_masked_edges[node_with_masked_edges] = True

        mask_both_node_edge = mask_x & mask_node_with_masked_edges
        mask_sole_node = mask_x & (~mask_both_node_edge)
        mask_sole_edge = mask_node_with_masked_edges & (~mask_both_node_edge)

        return x_m, ei_m, mask_both_node_edge, mask_sole_node, mask_sole_edge

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        assert self.config.training.scheduler_config is not None
        lrdec_1 = self.config.training.scheduler_config["lrdec_1"]
        lrdec_2 = self.config.training.scheduler_config["lrdec_2"]

        def comp_lr_factor(epoch: int) -> float:
            return lrdec_1 ** (epoch / lrdec_2)

        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=comp_lr_factor,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class GAE(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        predictor_config: dict[str, Any],
        emb_1_dim: int,
        emb_2_dim: int,
    ):
        super().__init__()
        self.encoder = GNNEncoder(**encoder_config)

        channel_list = [
            self.encoder.out_channels,
            predictor_config.pop("hidden_channels"),
        ]
        self.predictor_1 = MLP(channel_list=channel_list + [emb_1_dim], **predictor_config)
        self.predictor_2 = MLP(channel_list=channel_list + [emb_2_dim], **predictor_config)
        self.predictor_12 = MLP(
            channel_list=channel_list + [emb_1_dim + emb_2_dim],
            **predictor_config,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encode(x, edge_index)
        recon_embs = self.decode_all(z)
        return recon_embs

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x=x, edge_index=edge_index)

    def decode_all(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        recon_emb_1 = self.predictor_1(z)
        recon_emb_2 = self.predictor_2(z)
        recon_12 = self.predictor_12(z)
        return recon_emb_1, recon_emb_2, recon_12


class GiGaMAELoss(nn.Module):
    def __init__(
        self,
        tau: float,
        l1_e: float,
        l2_e: float,
        l12_e: float,
        l1_f: float,
        l2_f: float,
        l12_f: float,
        l1_b: float,
        l2_b: float,
        l12_b: float,
    ):
        super().__init__()
        self.l1_e = l1_e
        self.l2_e = l2_e
        self.l12_e = l12_e
        self.l1_f = l1_f
        self.l2_f = l2_f
        self.l12_f = l12_f
        self.l1_b = l1_b
        self.l2_b = l2_b
        self.l12_b = l12_b
        self.tau = tau

    def forward(
        self,
        emb_1: Tensor,
        emb_2: Tensor,
        recon_emb_1: Tensor,
        recon_emb_2: Tensor,
        recon_12: Tensor,
        mask_sole_node: Tensor,
        mask_sole_edge: Tensor,
        mask_both_node_edge: Tensor,
    ) -> Tensor:
        loss1_f = self.semi_loss(emb_1[mask_sole_node], recon_emb_1[mask_sole_node])
        loss1_e = self.semi_loss(emb_1[mask_sole_edge], recon_emb_1[mask_sole_edge])
        loss1_both = self.semi_loss(emb_1[mask_both_node_edge], recon_emb_1[mask_both_node_edge])

        loss2_f = self.semi_loss(emb_2[mask_sole_node], recon_emb_2[mask_sole_node])
        loss2_e = self.semi_loss(emb_2[mask_sole_edge], recon_emb_2[mask_sole_edge])
        loss2_both = self.semi_loss(emb_2[mask_both_node_edge], recon_emb_2[mask_both_node_edge])

        loss12_f = self.semi_loss(
            torch.cat((emb_1, emb_2), dim=1)[mask_sole_node], recon_12[mask_sole_node]
        )
        loss12_e = self.semi_loss(
            torch.cat((emb_1, emb_2), dim=1)[mask_sole_edge], recon_12[mask_sole_edge]
        )
        loss12_both = self.semi_loss(
            torch.cat((emb_1, emb_2), dim=1)[mask_both_node_edge], recon_12[mask_both_node_edge]
        )

        loss_e = self.l1_e * loss1_e + self.l2_e * loss2_e + self.l12_e * loss12_e
        loss_f = self.l1_f * loss1_f + self.l2_f * loss2_f + self.l12_f * loss12_f
        loss_both = self.l1_b * loss1_both + self.l2_b * loss2_both + self.l12_b * loss12_both

        info_loss = loss_e.mean() + loss_f.mean() + loss_both.mean()

        return info_loss

    def semi_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        # fixes the problem when there are sole_nodes/edges masked
        if not (z1.numel() and z2.numel()):
            return torch.tensor(0.0)

        refl_sim = torch.exp(self.similarity(z1, z1) / self.tau)

        between_sim = torch.exp(self.similarity(z1, z2) / self.tau)

        loss = -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

        return loss

    @staticmethod
    def similarity(z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


def svd_flip(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def postprocess(embs: Tensor, ratio: float) -> Tensor:
    # PCA
    mu = torch.mean(embs, dim=0, keepdim=True)
    X = embs - mu
    U, S, V = torch.svd(X)
    U, Vt = svd_flip(U, V)
    accumulate, sum_S = 0.0, sum(S.detach().cpu().tolist())
    for idx, s in enumerate(S.detach().cpu().tolist(), 1):
        accumulate += s / sum_S
        if accumulate > ratio:
            break
    X = torch.mm(X, Vt[:idx].T)

    # whitening
    u, s, vt = torch.svd(torch.mm(X.T, X) / (X.shape[0] - 1.0))
    W = torch.mm(u, torch.diag(1.0 / torch.sqrt(s)))
    X = torch.mm(X, W)
    return X


def pca(embs: Tensor, ratio: float) -> Tensor:
    if embs.shape[0] > embs.shape[1]:
        pca_emb = postprocess(embs, ratio)
    else:
        embs1 = embs[:, 0 : embs.shape[1] // 2]
        embs2 = embs[:, embs.shape[1] // 2 :]

        pca_emb1 = postprocess(embs1, ratio)
        pca_emb2 = postprocess(embs2, ratio)

        pca_emb = torch.cat((pca_emb1, pca_emb2), 1)

    return pca_emb

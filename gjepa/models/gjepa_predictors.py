from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

from gjepa.config import T_gjepa_predictor
from gjepa.models.gjepa_extractors import ExtractorOutput


class GraphGJEPAPredictor(nn.Module, ABC):
    def __init__(self, hidden_dim: int, pos_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.mask_token = nn.Parameter(data=torch.zeros(1, hidden_dim))
        self.pos_proj = nn.Linear(pos_dim, hidden_dim)

    @abstractmethod
    def forward(
        self, context_out: ExtractorOutput, target_outs: list[ExtractorOutput]
    ) -> tuple[Tensor, Tensor]:
        """Implements target prediction step of GJEPA method."""

    @staticmethod
    @abstractmethod
    def _get_type_name_to_predictor_cls_mapping() -> dict[
        T_gjepa_predictor, "GraphGJEPAPredictor"
    ]:
        ...

    @classmethod
    def from_config(
        cls, predictor_type: T_gjepa_predictor, predictor_kwargs: dict[str, Any]
    ) -> "GraphGJEPAPredictor":
        PREDICTORS = cls._get_type_name_to_predictor_cls_mapping()

        return PREDICTORS[predictor_type](**predictor_kwargs)

class GJEPAPredictor(GraphGJEPAPredictor):
    @staticmethod
    def _get_type_name_to_predictor_cls_mapping() -> dict[
        T_gjepa_predictor, "GJEPAPredictor"
    ]:
        return {
            "mlp": MLPGJEPAPredictor,
            "transformer": TransformerGJEPAPredictor,
        }

class MLPGJEPAPredictor(GJEPAPredictor):
    def __init__(
        self,
        pos_dim: int,
        hidden_dim: int,
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(pos_dim=pos_dim, hidden_dim=hidden_dim)
        self.mlp = MLP(in_channels=2 * self.hidden_dim, **mlp_kwargs, plain_last=True)

    def forward(
        self,
        context_out: ExtractorOutput,
        target_outs: list[ExtractorOutput],
    ) -> tuple[Tensor, Tensor]:
        """Predicts embeddings based on context and mask tokens.

        context_out.z == (batch_size, dim)
        context_out.pos == (batch_size, pos_dim)
        context_out.mask == None

        target_out.z == (batch_size, num_targets, dim)
        - target_outs[i].z == (batch_size, dim)
        target_out.pos == (batch_size, num_targets, pos_dim)
        - target_outs[i].pos == (batch_size, pos_dim)
        target_outs[i].mask == None
        """
        assert context_out.z.dim() == 2
        assert all(target.z.dim() == 2 for target in target_outs)
        assert all(context_out.z.shape[0] == target.z.shape[0] for target in target_outs)

        pos_target = self._prepare_pos_target(target_outs)

        z_context = self._prepare_context_embedding(context_out, num_targets=len(target_outs))

        x = torch.cat([z_context, pos_target], dim=-1)

        pred_target = self.mlp(x.flatten(end_dim=1))

        z_target = torch.cat([target.z for target in target_outs], dim=0)

        return pred_target, z_target

    def _prepare_pos_target(self, target_outs: list[ExtractorOutput]) -> Tensor:
        pos_target = torch.stack([target.pos for target in target_outs], dim=1)
        batch_size, num_targets, _ = pos_target.shape
        pos_target = self.pos_proj(pos_target)
        pos_target = pos_target + self.mask_token.repeat(batch_size, num_targets, 1)
        return pos_target

    def _prepare_context_embedding(self, context_out: ExtractorOutput, num_targets: int) -> Tensor:
        pos_context = self.pos_proj(context_out.pos)
        z_context = context_out.z + pos_context
        z_context = z_context.unsqueeze(1).repeat(1, num_targets, 1)

        return z_context


class TransformerGJEPAPredictor(GJEPAPredictor):
    def __init__(
        self,
        pos_dim: int,
        hidden_dim: int,
        num_layers: int,
        transformer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(pos_dim=pos_dim, hidden_dim=hidden_dim)
        self.num_heads = transformer_kwargs["nhead"]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            batch_first=True,
            **transformer_kwargs,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        context_out: ExtractorOutput,
        target_outs: list[ExtractorOutput],
    ) -> tuple[Tensor, Tensor]:
        """Predicts embeddings based on context and mask tokens.

        context_out.z == (batch_size, PADDING_DIM_CTX, dim)
        context_out.pos == (batch_size, PADDING_DIM_CTX, pos_dim)
        context_out.mask == (batch_size, PADDING_DIM_CTX)

        target_out.z == (batch_size, num_targets, PADDING_DIM_T, dim)
        - target_outs[i].z == (batch_size, PADDING_DIM_T, dim)
        target_out.pos == (batch_size, num_targets, PADDING_DIM_T, pos_dim)
        - target_outs[i].pos == (batch_size, PADDING_DIM_T, pos_dim)
        target_outs[i].mask == (batch_size, PADDING_DIM_T)
        """
        assert context_out.z.dim() == 3
        assert all(target.z.dim() == 3 for target in target_outs)
        assert all(context_out.z.shape[0] == target.z.shape[0] for target in target_outs)

        pos_target = self._prepare_pos_target(target_outs)

        z_context = self._prepare_context_embedding(context_out, num_targets=len(target_outs))

        # Now:
        # pos_target.shape: (batch_size, num_targets, PADDING_DIM_T, dim)
        # z_context.shape: (batch_size, num_targets, PADDING_DIM_CTX, dim)

        x = torch.cat([z_context, pos_target], dim=2)
        mask, mask_t = self._prepare_mask(context_out, target_outs)

        # Now:
        # x.shape: (batch_size, num_targets, PADDING_DIM_CTX + PADDING_DIM_T, dim)
        # mask.shape: (batch_size, num_targets, PADDING_DIM_CTX + PADDING_DIM_T)

        x = x.flatten(end_dim=1)
        mask = mask.flatten(end_dim=1)
        mask_t = mask_t.flatten(end_dim=1)

        # Now:
        # x.shape: (batch_size * num_targets, PADDING_DIM_CTX + PADDING_DIM_T, dim)
        # mask.shape: (batch_size * num_targets, PADDING_DIM_CTX + PADDING_DIM_T)

        attention_mask = torch.bmm(mask.float().unsqueeze(-1), mask.float().unsqueeze(1))
        attention_mask = attention_mask.repeat(self.num_heads, 1, 1)

        pred_target = self.transformer_encoder(
            src=x,
            mask=attention_mask,
            src_key_padding_mask=(~mask).float(),
        )

        pred_target = pred_target[:, z_context.size(2) :]
        z_target = torch.stack([target.z for target in target_outs], dim=1).flatten(end_dim=1)

        pred_target = pred_target[mask_t]
        z_target = z_target[mask_t]

        return pred_target, z_target

    def _prepare_pos_target(self, target_outs: list[ExtractorOutput]) -> Tensor:
        pos_target = torch.stack([target.pos for target in target_outs], dim=1)
        batch_size, num_targets, padding_dim_t, _ = pos_target.shape
        pos_target = self.pos_proj(pos_target)
        pos_target = pos_target + self.mask_token.repeat(batch_size, num_targets, padding_dim_t, 1)
        return pos_target

    def _prepare_context_embedding(self, context_out: ExtractorOutput, num_targets: int) -> Tensor:
        pos_context = self.pos_proj(context_out.pos)
        z_context = context_out.z + pos_context
        z_context = z_context.unsqueeze(dim=1).repeat(1, num_targets, 1, 1)

        return z_context

    def _prepare_mask(
        self, context_out: ExtractorOutput, target_outs: list[ExtractorOutput]
    ) -> tuple[Tensor, Tensor]:
        # todo: consider padding token instead of zeros for the input
        mask_t = torch.stack([t.mask for t in target_outs], dim=1)  # type: ignore[misc]
        num_targets = len(target_outs)

        mask_ctx = context_out.mask.unsqueeze(dim=1).repeat(1, num_targets, 1)  # type: ignore

        return torch.cat([mask_ctx, mask_t], dim=2), mask_t

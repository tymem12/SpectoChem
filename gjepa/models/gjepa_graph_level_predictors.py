from typing import Any

from torch import nn

import torch
from torch import Tensor
from torch_geometric.nn import MLP

from gjepa.config import T_gjepa_predictor

from gjepa.models.gjepa_predictors import GraphGJEPAPredictor

from gjepa.models.gjepa_graph_level_extractors import GraphLevelExtractorOutput

class GJEPAGraphLevelPredictor(GraphGJEPAPredictor):
    @staticmethod
    def _get_type_name_to_predictor_cls_mapping() -> dict[
        T_gjepa_predictor, "GJEPAGraphLevelPredictor"
    ]:
        return {
            "mlp": MLPGraphLevelGJEPAPredictor,
            "transformer": TransformerGraphLevelGJEPAPredictor,
        }

class MLPGraphLevelGJEPAPredictor(GJEPAGraphLevelPredictor):
    def __init__(
        self,
        pos_dim: int,
        hidden_dim: int,
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, pos_dim=pos_dim)

        self.mlp = MLP(
            in_channels=2 * hidden_dim,
            **mlp_kwargs,
            plain_last=True
        )

    def forward(
        self,
        context_out: GraphLevelExtractorOutput,
        target_outs: list[GraphLevelExtractorOutput],
    ) -> tuple[Tensor, Tensor]:
        """
        Graph-level prediction.

        context_out.z: (batch_size, hidden_dim)
        context_out.pos: (batch_size, pos_dim)

        target_outs[i].z: (batch_size, hidden_dim)
        target_outs[i].pos: (batch_size, pos_dim)
        """
        assert context_out.z.dim() == 2
        assert all(target.z.dim() == 2 for target in target_outs)

        pos_target = self._prepare_pos_target(target_outs)

        # prepare context embedding
        z_context = self._prepare_context_embedding(context_out, num_targets=len(target_outs))

        x = torch.cat([z_context, pos_target], dim=-1)

        # flatten batch and num_targets
        pred_target = self.mlp(x.flatten(end_dim=1))

        # ground-truth target graph embeddings
        z_target = torch.cat([target.z for target in target_outs], dim=0)

        return pred_target, z_target

    def _prepare_pos_target(self, target_outs: list[GraphLevelExtractorOutput]) -> Tensor:
        pos_target = torch.stack([target.pos for target in target_outs], dim=1)

        batch_size, num_targets, _ = pos_target.shape
        pos_target = self.pos_proj(pos_target)

        pos_target = pos_target + self.mask_token.repeat(batch_size, num_targets, 1)
        return pos_target

    def _prepare_context_embedding(self, context_out: GraphLevelExtractorOutput, num_targets: int) -> Tensor:
        pos_context = self.pos_proj(context_out.pos)
        z_context = context_out.z + pos_context
        z_context = z_context.unsqueeze(1).repeat(1, num_targets, 1)
        return z_context



class TransformerGraphLevelGJEPAPredictor(GJEPAGraphLevelPredictor):
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
            d_model=hidden_dim,
            batch_first=True,
            **transformer_kwargs,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        context_out: GraphLevelExtractorOutput,
        target_outs: list[GraphLevelExtractorOutput],
    ) -> tuple[Tensor, Tensor]:
        """
        Graph-level prediction with transformer.

        context_out.z: (batch_size, hidden_dim)
        context_out.pos: (batch_size, pos_dim)

        target_outs[i].z: (batch_size, hidden_dim)
        target_outs[i].pos: (batch_size, pos_dim)
        """
        assert context_out.z.dim() == 2
        assert all(target.z.dim() == 2 for target in target_outs)

        pos_target = self._prepare_pos_target(target_outs)

        z_context = self._prepare_context_embedding(context_out, num_targets=len(target_outs))

        x = torch.cat([z_context, pos_target], dim=1)

        pred = self.transformer_encoder(x)

        # take predicted target slice
        pred_target = pred[:, z_context.size(1):]

        z_target = torch.stack([target.z for target in target_outs], dim=1)

        return pred_target.flatten(end_dim=1), z_target.flatten(end_dim=1)

    def _prepare_pos_target(self, target_outs: list[GraphLevelExtractorOutput]) -> Tensor:
        pos_target = torch.stack([target.pos for target in target_outs], dim=1)
        pos_target = self.pos_proj(pos_target)
        pos_target = pos_target + self.mask_token
        return pos_target

    def _prepare_context_embedding(self, context_out: GraphLevelExtractorOutput, num_targets: int) -> Tensor:
        pos_context = self.pos_proj(context_out.pos)
        z_context = context_out.z + pos_context
        z_context = z_context.unsqueeze(1).repeat(1, num_targets, 1)
        return z_context

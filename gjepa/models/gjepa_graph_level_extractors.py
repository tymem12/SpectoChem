from typing import Any

import torch

from torch import nn
from torch_geometric.data import Batch, Data

from gjepa.config import T_graph_level_extractor

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from gjepa.models.gjepa_extractors import GraphExtractor, GraphExtractorOutput

class GraphLevelExtractorOutput(GraphExtractorOutput):
    pass

class GraphLevelExtractor(GraphExtractor[GraphLevelExtractorOutput, T_graph_level_extractor]):
    @staticmethod
    def _get_type_name_to_extractor_cls_mapping() -> dict[
        T_graph_level_extractor, "GraphLevelExtractor"
    ]:
        return {
            "mean_pool": MeanPoolingGraphLevelExtractor,
            "sum_pool": SumPoolingGraphLevelExtractor,
            "max_pool": MaxPoolingGraphLevelExtractor
        }

class PoolingGraphLevelExtractor(GraphLevelExtractor):
    def __init__(self, encoder: nn.Module, pooling_fn):
        """
        Graph-level extractor that pools node embeddings into graph embeddings.

        Args:
            encoder: GNN encoder (operates on nodes)
            pooling_fn
        """
        super().__init__(encoder)

        self.pool = pooling_fn

    def forward(self, batch: Batch, **kwargs: Any) -> GraphLevelExtractorOutput:
        pos_enc = batch.positional_encoding

        # node embeddings from encoder
        z_all = self.encoder(
            batch
        )

        if hasattr(batch, "target_mask"):
            target_mask = batch.target_mask.view(-1)

            z_all = z_all[target_mask]
            pos_enc = pos_enc[target_mask]

            batch = batch.batch[target_mask]
        else:
            batch = batch.batch

        # pool to graph-level embeddings
        z_graph = self.pool(z_all, batch)
        pos_graph = self.pool(pos_enc, batch)

        return GraphLevelExtractorOutput(
            z=z_graph,
            pos=pos_graph
        )

class MeanPoolingGraphLevelExtractor(PoolingGraphLevelExtractor):
    def __init__(self, encoder: nn.Module):
        super().__init__(encoder, global_mean_pool)

class SumPoolingGraphLevelExtractor(PoolingGraphLevelExtractor):
    def __init__(self, encoder: nn.Module):
        super().__init__(encoder, global_add_pool)

class MaxPoolingGraphLevelExtractor(PoolingGraphLevelExtractor):
    def __init__(self, encoder: nn.Module):
        super().__init__(encoder, global_max_pool)

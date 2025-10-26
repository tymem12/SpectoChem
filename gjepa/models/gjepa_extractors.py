from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar

from torch import Tensor, nn
from torch_geometric.data import Batch

from gjepa.config import T_extractor

class GraphExtractorOutput(NamedTuple):
    z: Tensor
    pos: Tensor

T = TypeVar("T", bound=GraphExtractorOutput)
U = TypeVar("U", bound=str)

class GraphExtractor(nn.Module, Generic[T, U], ABC):
    def __init__(self, encoder: nn.Module):
        super().__init__()

        self.encoder = encoder

    @abstractmethod
    def forward(self, batch: Batch, **kwargs: Any) -> T:
        pass

    @staticmethod
    @abstractmethod
    def _get_type_name_to_extractor_cls_mapping() -> dict[
        U, "GraphExtractor"
    ]:
        ...

    @classmethod
    def from_config(cls, extractor_type: U, encoder: nn.Module) -> "GraphExtractor":
        EXTRACTORS = cls._get_type_name_to_extractor_cls_mapping()

        return EXTRACTORS[extractor_type](encoder)

class ExtractorOutput(GraphExtractorOutput):
    mask: Tensor | None

class Extractor(GraphExtractor[ExtractorOutput, T_extractor]):
    @staticmethod
    def _get_type_name_to_extractor_cls_mapping() -> dict[
        T_extractor, "Extractor"
    ]:
        return {
            "node": RootNodeExtractor,
            "subgraph": SubgraphExtractor,
        }

class RootNodeExtractor(Extractor):
    def forward(self, batch: Batch, **kwargs: Any) -> ExtractorOutput:
        """Due to empty graphs included in the batch by the sampler, we remove
        repeating elements from the `ptr`.
        """
        z_all = self.encoder(x=batch.x, edge_index=batch.edge_index)
        root_node_mask = batch.ptr[:-1]

        return ExtractorOutput(
            z=z_all[root_node_mask],
            pos=batch.pos[root_node_mask],
            mask=None,
        )


class SubgraphExtractor(Extractor):
    def forward(self, batch: Batch, **kwargs: Any) -> ExtractorOutput:
        z_all = self.encoder(x=batch.x, edge_index=batch.edge_index)

        max_num_nodes = kwargs.get("max_num_nodes")
        z, z_mask = to_dense_batch(x=z_all, batch=batch.batch, max_num_nodes=max_num_nodes)
        pos, pos_mask = to_dense_batch(x=batch.pos, batch=batch.batch, max_num_nodes=max_num_nodes)

        assert (pos_mask == z_mask).all(), "Embedding and pos_encoding masks should be equal"

        return ExtractorOutput(
            z=z,
            pos=pos,
            mask=z_mask,
        )

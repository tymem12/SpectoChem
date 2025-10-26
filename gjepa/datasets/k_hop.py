import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from gjepa.config import T_gjepa_overlap_strategy


class KHopDataset(Dataset):
    def __init__(
        self,
        k_hop_subgraphs: list[Data],
        num_targets: int,
        num_hops: int,
        similarity_matrix_file: str | None,
        context_target_overlap_strategy: T_gjepa_overlap_strategy,
    ):
        super().__init__()
        self.k_hop_subgraphs = k_hop_subgraphs
        self.num_targets = num_targets
        self.num_hops = num_hops

        self.num_graphs = len(k_hop_subgraphs)

        self.empty_graph = Data(
            x=torch.empty(0),
            edge_index=torch.empty(0, dtype=torch.long),
            pos=torch.empty(0),
        )

        self.similarity_matrix: Tensor | None = None

        if similarity_matrix_file is not None:
            self.similarity_matrix = self.init_similarity_matrix(
                input_file=similarity_matrix_file,
            )

        self.context_target_overlap_strategy = context_target_overlap_strategy

    def init_similarity_matrix(self, input_file: str) -> Tensor:
        sm = torch.load(f=input_file)

        # Prevent context to be sampled as its own target
        sm[torch.eye(self.num_graphs).bool()] = 0

        # Remove NaN values
        sm[torch.isnan(sm)] = 1e-6

        return sm

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, index: int) -> dict[str, Data | list[Data]] | None:
        """Returns context and targets as {context: Data, targets: list[Data]}"""
        if self.similarity_matrix is not None:
            weights = self.similarity_matrix[index]
        else:
            weights = torch.ones(self.num_graphs)
            weights[index] = 0

        target_indices = torch.multinomial(
            input=weights,
            num_samples=self.num_targets,
            replacement=False,
        )

        targets, target_n_ids = [], []
        for t_idx in target_indices.tolist():
            t_graph = self.k_hop_subgraphs[t_idx].clone()
            del t_graph.y

            target_n_ids.append(t_graph.n_id)
            del t_graph.n_id
            targets.append(t_graph)

        if self.context_target_overlap_strategy == "ignore":
            ctx_graph = self.k_hop_subgraphs[index]
        elif self.context_target_overlap_strategy == "remove":
            ctx_graph = self._remove_target_nodes_from_ctx(
                ctx_graph=self.k_hop_subgraphs[index],
                target_n_ids=target_n_ids,
            )
        elif self.context_target_overlap_strategy == "mask":
            ctx_graph = self._mask_target_nodes_from_ctx(
                ctx_graph=self.k_hop_subgraphs[index],
                target_n_ids=target_n_ids,
            )

        if ctx_graph.num_edges == 0:
            return None

        return {
            "context": ctx_graph,
            "targets": targets,
        }

    def _remove_target_nodes_from_ctx(self, ctx_graph: Data, target_n_ids: list[Tensor]) -> Data:
        target_n_ids_t = torch.cat(target_n_ids).unique()

        non_intersecting_edge_mask = torch.isin(
            ctx_graph.n_id[ctx_graph.edge_index],
            target_n_ids_t,
            invert=True,
        ).all(dim=0)

        if non_intersecting_edge_mask.sum() == 0:
            return self.empty_graph

        node_mask, edge_index, mapping, _ = k_hop_subgraph(
            node_idx=0,
            num_hops=self.num_hops,
            edge_index=ctx_graph.edge_index[:, non_intersecting_edge_mask],
            relabel_nodes=True,
        )
        assert mapping[0] == 0

        return Data(
            x=ctx_graph.x[node_mask],
            edge_index=edge_index,
            pos=ctx_graph.pos[node_mask],
        )

    def _mask_target_nodes_from_ctx(self, ctx_graph: Data, target_n_ids: list[Tensor]) -> Data:
        target_n_ids_t = torch.cat(target_n_ids).unique()

        node_mask = torch.isin(ctx_graph.n_id, target_n_ids_t)
        x_masked = ctx_graph.x
        x_masked[node_mask] = 0

        return Data(
            x=x_masked,
            edge_index=ctx_graph.edge_index,
            pos=ctx_graph.pos,
            target_node_mask=node_mask,
        )

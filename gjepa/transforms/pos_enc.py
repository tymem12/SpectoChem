from typing import Literal

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, BaseTransform
from torch_geometric.transforms.add_positional_encoding import add_node_attr
from torch_geometric.utils import degree, get_laplacian, to_networkx, to_scipy_sparse_matrix


class AnchorBasedPE(BaseTransform):
    def __init__(self, num_anchors: int, anchor_sampling: Literal["random", "degree"]):
        self.num_anchors = num_anchors
        self.anchor_sampling = anchor_sampling

    def __call__(self, data: Data) -> Data:
        if self.anchor_sampling == "random":
            anchor_nodes = torch.randperm(data.num_nodes)[: self.num_anchors]
        elif self.anchor_sampling == "degree":
            degs = degree(data.edge_index[0])
            anchor_nodes = torch.multinomial(
                input=degs, num_samples=self.num_anchors, replacement=False
            )
        else:
            raise ValueError(f"Invalid anchor_sampling: '{self.anchor_sampling}'")

        graph = to_networkx(data)
        pos_enc = torch.zeros(data.num_nodes, self.num_anchors, dtype=torch.float)
        for ith_anchor, anchor_id in enumerate(anchor_nodes.tolist()):
            path_lengths = nx.shortest_path_length(graph, target=anchor_id)
            for node_id in range(data.num_nodes):
                # returns 0.0 for path from node to itself, -1 for non-reachable nodes
                pos_enc[node_id][ith_anchor] = path_lengths.get(node_id, -1)

        data.pos = pos_enc

        return data


class DeterministicAddLaplacianEigenvectorPE(AddLaplacianEigenvectorPE):
    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_vals, eig_vecs = np.linalg.eig(L.toarray())

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])

        pe = torch.from_numpy(eig_vecs[:, 1 : self.k + 1]).float()
        sign = -1 + 2 * torch.randint(0, 2, (self.k,))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data


class AllOnesPosencs(BaseTransform):
    """Dummy positional encodings for testing purposes."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, data: Data) -> Data:
        data.pos = torch.ones(data.x.size(0), self.dim, dtype=torch.float)
        return data

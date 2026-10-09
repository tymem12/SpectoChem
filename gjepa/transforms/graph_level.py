from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from gjepa.utils.graph_level import edge_index_from_smiles, calc_edge_index, calc_edge_weight
from torch_cluster import radius_graph


class ToFloat(BaseTransform):
    def __call__(self, data: Data) -> Data:
        data.x = data.x.float()
        return data



class SelectTargets(BaseTransform):
    def __init__(self, num_targets: int = 12):
        self.num_targets = num_targets

    def __call__(self, data):
        data.y = data.y[..., :self.num_targets]
        return data


class AddEdgesAndDistances(object):
    def __init__(self, cutoff=10.0):
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index = radius_graph(data.pos, r=self.cutoff, loop=False, max_num_neighbors=64)
        edge_weight = calc_edge_weight(data.pos, edge_index)
        data.edge_index = edge_index
        data.edge_weight = edge_weight          # or: data.edge_attr = edge_weight.view(-1,1)
        return data
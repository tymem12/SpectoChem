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



class QM9EnergyToMeV(BaseTransform):
    """
    Transform: przelicza wszystkie energie w QM9 z eV na meV (×1000).
    Indeksy w y:
      2 ε_HOMO [eV]
      3 ε_LUMO [eV]
      4 Δε [eV]
      6 ZPVE [eV]
      7 U0 [eV]
      8 U  [eV]
      9 H  [eV]
      10 G [eV]
    """

    ENERGY_IDX = [2, 3, 4, 6, 7, 8, 9, 10]

    def __init__(self, y_key: str = "y"):
        self.y_key = y_key

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, self.y_key):
            return data

        y = getattr(data, self.y_key).clone()

        if y.ndim == 1:
            y[self.ENERGY_IDX] *= 1000.0
        elif y.ndim == 2 and y.size(-1) >= 12:
            y[:, self.ENERGY_IDX] *= 1000.0

        setattr(data, self.y_key, y)
        return data


class AddEdgesAndDistances(object):
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index = radius_graph(data.pos, r=self.cutoff, loop=False, max_num_neighbors=64)
        edge_weight = calc_edge_weight(data.pos, edge_index)
        data.edge_index = edge_index
        data.edge_weight = edge_weight          # or: data.edge_attr = edge_weight.view(-1,1)
        return data
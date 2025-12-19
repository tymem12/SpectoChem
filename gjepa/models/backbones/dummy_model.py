import torch
from torch import nn
from torch_geometric.nn import radius_graph, global_add_pool
from torch_geometric.data import Data


class DummyModel(nn.Module):

    def __init__(self,
                 hidden_channels: int = 128,
                 **kwargs):
        super().__init__()
        self.handles_pos_encoding = True
        self.kwargs = kwargs
        self.hidden_channels = hidden_channels
        self.out_channels = self.hidden_channels


    def forward(self, batch: Data):
        embeding = batch.representation
        return embeding

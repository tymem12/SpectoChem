import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData


class ClusterDataset(Dataset):
    def __init__(self, cluster_data: ClusterData, num_targets: int):
        self.cluster_data = cluster_data
        self.num_targets = num_targets

        # clustering provides some clusters being empty
        self.index = torch.tensor([i for i, d in enumerate(self.cluster_data) if len(d.x) > 0])

    def __len__(self) -> int:
        return self.index.numel()

    def __getitem__(self, item: int) -> dict[str, Data | list[Data]]:
        target_candidate_idx = torch.cat([self.index[:item], self.index[item + 1 :]])
        target_idx = target_candidate_idx[
            torch.randperm(len(target_candidate_idx))[: self.num_targets]
        ]
        targets = [self.cluster_data[t_idx] for t_idx in target_idx]

        cluster_idx = self.index[item]
        return {"context": self.cluster_data[cluster_idx], "targets": targets}

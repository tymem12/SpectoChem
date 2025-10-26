import os
import warnings
from pathlib import Path

import abc

import torch
import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import DataTensorAttr
from torch_geometric.datasets import (
    Amazon,
    Coauthor,
    HeterophilousGraphDataset,
    Planetoid,
    WikiCS,
)
from torch_geometric.loader import ClusterData, NeighborLoader
from torch_geometric.transforms import BaseTransform

from gjepa.config import (
    GraphDatasetConfig,
    DatasetConfig,
    T_dataloader_mode,
    T_gjepa_overlap_strategy,
)
from gjepa.datasets.cluster import ClusterDataset
from gjepa.datasets.collators import EmptyGraphFilteringCollater
from gjepa.datasets.k_hop import KHopDataset
from gjepa.params import seeds
from gjepa.utils.pyg import create_transform

warnings.filterwarnings("ignore", ".*does not have many workers.*")

NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset_config: GraphDatasetConfig, batch_size: int):
        super().__init__()

        self.config = dataset_config
        self.batch_size = batch_size

    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader:
        pass

    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass


class KHopDatamodule(GraphDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        batch_size: int,
        dataloader_mode: T_dataloader_mode,
        pos_enc_path: Path | None = None,
    ):
        super().__init__(dataset_config, batch_size)

        self.dataloader_mode = dataloader_mode
        self.pos_enc_path = pos_enc_path

        self.dataset: Data | None = None

    def setup(self, stage: str) -> None:
        self.dataset = self._load_dataset()

    def train_dataloader(self) -> NeighborLoader:
        assert self.dataset is not None
        if self.dataloader_mode == "transductive":
            mask = None
        else:
            mask = self.dataset.train_mask
        return self._create_loader(mask=mask, shuffle=True)

    def train_inference_dataloader(self) -> NeighborLoader:
        assert self.dataset is not None
        """DataLoader used for full-dataset evaluation after finishing training epoch."""
        return self._create_loader(self.dataset.train_mask, shuffle=False)

    def val_dataloader(self) -> NeighborLoader:
        assert self.dataset is not None
        return self._create_loader(self.dataset.val_mask, shuffle=False)

    def test_dataloader(self) -> NeighborLoader:
        assert self.dataset is not None
        return self._create_loader(self.dataset.test_mask, shuffle=False)

    def _create_loader(self, mask: Tensor | None, shuffle: bool) -> NeighborLoader:
        assert self.dataset is not None
        return NeighborLoader(
            data=self.dataset,
            input_nodes=mask,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            disjoint=True,
        )

    def _load_dataset(self) -> Data:
        dataset = load_graph(
            root_dir=self.config.root_dir,
            name=self.config.name,
            pre_transform=create_transform(self.config.pre_transforms),
            transform=create_transform(self.config.transforms),
        )

        if self.pos_enc_path is not None:
            pos = torch.load(self.pos_enc_path)
            dataset.pos = pos

        return dataset


class GJEPANodeLevelDataModule(KHopDatamodule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        batch_size: int,
        dataloader_mode: T_dataloader_mode,
        pos_enc_path: Path,
        num_targets: int,
        similarity_matrix_file: str | None,
        context_target_overlap_strategy: T_gjepa_overlap_strategy,
    ):
        super().__init__(dataset_config, batch_size, dataloader_mode, pos_enc_path)
        self.num_targets = num_targets
        self.similarity_matrix_file = similarity_matrix_file
        self.context_target_overlap_strategy = context_target_overlap_strategy

    def train_dataloader(self) -> DataLoader:
        """Creates custom train loader with context and target sampling on each __get_item__().

        NOTE: Sampling of kHop dataset should be performed in each epoch, so one should set
        pl.Trainer(reload_dataloaders_every_n_epochs=1)

        """
        assert self.dataset is not None
        if self.dataloader_mode == "transductive":
            mask = None
        else:
            mask = self.dataset.train_mask
        k_hop_dataset = self._sample_k_hop_dataset(mask=mask)
        return DataLoader(
            dataset=k_hop_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=EmptyGraphFilteringCollater(),
            num_workers=NUM_WORKERS,
        )

    def _sample_k_hop_dataset(self, mask: Tensor | None) -> KHopDataset:
        sampling_loader = NeighborLoader(
            data=self.dataset,
            input_nodes=mask,
            num_neighbors=self.config.num_neighbors,
            shuffle=False,
            batch_size=1,
            num_workers=0,
        )

        k_hop_graphs: list[Data] = []

        for graph in sampling_loader:
            data = Data(
                x=graph.x,
                edge_index=graph.edge_index,
                y=graph.y,
                pos=graph.pos,
                n_id=graph.n_id,
            )
            k_hop_graphs.append(data)

        return KHopDataset(
            k_hop_subgraphs=k_hop_graphs,
            num_targets=self.num_targets,
            num_hops=self.config.num_hops,
            similarity_matrix_file=self.similarity_matrix_file,
            context_target_overlap_strategy=self.context_target_overlap_strategy,
        )


class ClusterDatamodule(KHopDatamodule):
    """Datamodule with training items determined by clustering (val/test are k-hop)."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        batch_size: int,
        dataloader_mode: T_dataloader_mode,
        pos_enc_path: Path,
        num_targets: int,
        cache_dir: str | Path | None = None,
    ):
        super().__init__(dataset_config, batch_size, dataloader_mode, pos_enc_path)
        self.num_targets = num_targets
        self.cache_dir = cache_dir

        self.clustered_dataset: ClusterDataset | None = None

    def setup(self, stage: str) -> None:
        super().setup(stage)

        clustered_data = ClusterData(
            data=self.dataset,
            num_parts=self.config.num_partitions,
            save_dir=self.cache_dir,
        )
        self.clustered_dataset = ClusterDataset(
            cluster_data=clustered_data,
            num_targets=self.num_targets,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.clustered_dataset is not None
        return DataLoader(
            self.clustered_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=EmptyGraphFilteringCollater(),
        )


class SubgraphDatamodule(KHopDatamodule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        batch_size: int,
        dataloader_mode: T_dataloader_mode,
    ):
        super().__init__(dataset_config, batch_size, dataloader_mode)

    def _create_loader(self, mask: Tensor | None, shuffle: bool) -> NeighborLoader:
        return NeighborLoader(
            data=self.dataset,
            input_nodes=mask,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            disjoint=True,  # necessary for mapping nodes to their subgraphs
        )


def load_graph(
    root_dir: Path,
    name: str,
    transform: BaseTransform | None = None,
    pre_transform: BaseTransform | None = None,
) -> Data:
    if name in ("Cora", "CiteSeer", "PubMed"):
        data = Planetoid(
            root=root_dir,
            name=name,
            split="public",
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "WikiCS":
        root_dir = root_dir / "WikiCS"
        data = WikiCS(
            root=root_dir,
            is_undirected=True,
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "Amazon-CS":
        root_dir = root_dir / "Amazon-CS"
        data = Amazon(
            root=root_dir,
            name="computers",
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "Amazon-Photo":
        root_dir = root_dir / "Amazon-Photo"
        data = Amazon(
            root=root_dir,
            name="photo",
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "Coauthor-CS":
        root_dir = root_dir / "Coauthor-CS"
        data = Coauthor(
            root=root_dir,
            name="cs",
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "Coauthor-Physics":
        root_dir = root_dir / "Coauthor-Physics"
        data = Coauthor(
            root=root_dir,
            name="physics",
            transform=transform,
            pre_transform=pre_transform,
        )[0]
    elif name == "ogbn-arxiv":
        root_dir = root_dir / "ogbn-arxiv"
        with torch.serialization.safe_globals(
            [DataTensorAttr, torch_geometric.data.storage.GlobalStorage]
        ):
            dataset = PygNodePropPredDataset(
                root=root_dir,
                name="ogbn-arxiv",
                transform=transform,
                pre_transform=pre_transform,
            )
        split_idx = dataset.get_idx_split()

        data = dataset[0]

        def _prepare_mask(indices: Tensor) -> Tensor:
            mask = torch.zeros(data.num_nodes).bool()
            mask[indices] = True
            mask = mask.unsqueeze(dim=-1).repeat(1, len(seeds))
            return mask

        data.train_mask = _prepare_mask(split_idx["train"])
        data.val_mask = _prepare_mask(split_idx["valid"])
        data.test_mask = _prepare_mask(split_idx["test"])

        data.y = data.y.squeeze(dim=-1)
    elif name == "roman_empire":
        dataset = HeterophilousGraphDataset(
            root=root_dir,
            name="Roman-empire",
            transform=transform,
            pre_transform=pre_transform,
        )
        data = dataset[0]
        data.y = data.y.squeeze(dim=-1)
    elif name == "amazon_ratings":
        dataset = HeterophilousGraphDataset(
            root=root_dir,
            name="Amazon-ratings",
            transform=transform,
            pre_transform=pre_transform,
        )
        data = dataset[0]
        data.y = data.y.squeeze(dim=-1)
    elif name == "questions":
        dataset = HeterophilousGraphDataset(
            root=root_dir,
            name="Questions",
            transform=transform,
            pre_transform=pre_transform,
        )
        data = dataset[0]
        data.y = data.y.unsqueeze(dim=-1).float()
    else:
        raise ValueError(f"Invalid dataset name in config: {name}")

    mask_idx = seeds.index(torch.initial_seed())
    data.train_mask = data.train_mask[:, mask_idx]
    data.val_mask = data.val_mask[:, mask_idx]
    data.test_mask = data.test_mask[:, mask_idx]

    assert data.train_mask.dtype == torch.bool
    assert data.val_mask.dtype == torch.bool
    assert data.test_mask.dtype == torch.bool

    return data

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split, Subset
import os
from typing import Optional, Any
from collections.abc import Sequence

import random

import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Dataset, InMemoryDataset
from torch_geometric.datasets import ZINC, QM9, TUDataset
from torch_geometric.transforms import BaseTransform

from gjepa.utils.pyg import create_transform

from gjepa.config import GraphLevelDatasetConfig
from gjepa.datasets.node_level import GraphDataModule
from gjepa.datasets.cv_vis.custom_ds import SpectoDataset
from gjepa.utils.pos_encoding import attach_pe_to_dataset_inplace
from gjepa.utils.graph_level import split_dataset
import torch.nn.functional as F


def load_graph(
    root_dir: Path,
    name: str,
    transform: BaseTransform | None = None,
    pre_transform: BaseTransform | None = None,
    additional_loading_params: dict[str, Any] | None = None
) -> Dataset:
    kwargs = dict(root=root_dir, transform=transform, pre_transform=pre_transform) | additional_loading_params

    if name == "TMQM_SPECTO":
        dataset = SpectoDataset(**kwargs)
    else:
        raise ValueError(f"Invalid dataset name in config: {name!r}")

    return dataset

class GraphLevelDataModule(GraphDataModule):
    def __init__(
        self,
        dataset_config: GraphLevelDatasetConfig,
        batch_size: int,
        pos_enc_path: Path = None
    ):
        super().__init__(dataset_config, batch_size)
        self.config = dataset_config
        self.batch_size = batch_size
        self.pos_enc_path = pos_enc_path

        self.train_ds: Subset | None = None
        self.val_ds: Subset | None = None
        self.test_ds: Subset | None = None

        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

    def setup(self, stage: str) -> None:
        name = self.config.name

        split_ratios = self.config.split_ratios

        should_split = name != ZINC.__name__

        if should_split:
            if split_ratios is None:
                raise ValueError(f"{name!r} dataset requires `split_ratios` (e.g., `[0.8, 0.1]`)")

            if (
                not isinstance(split_ratios, Sequence)
                or len(split_ratios) != 2
                or any(r <= 0 for r in split_ratios)
                or sum(split_ratios) >= 1
            ):
                raise ValueError(
                    "`split_ratios` must be a sequence of two positive floats summing to less than 1.0"
                )
        elif split_ratios is not None:
            raise ValueError(
                f"{name!r} dataset uses official splits; `split_ratios` must be `None`."
            )
        dataset = load_graph(
            root_dir=self.config.root_dir,
            name=name,
            additional_loading_params=self.config.additional_loading_params,
            pre_transform=create_transform(self.config.pre_transforms),
            transform=create_transform(self.config.transforms),

        )



        if self.pos_enc_path is not None:
            attach_pe_to_dataset_inplace(dataset=dataset, pe_path=self.pos_enc_path)
        if should_split:
            self.train_ds, self.val_ds, self.test_ds = split_dataset(
                dataset, split_ratios
            )
        else:
            self.train_ds = Subset(dataset, dataset.split_indices["train"])
            self.val_ds   = Subset(dataset, dataset.split_indices["val"])
            self.test_ds  = Subset(dataset, dataset.split_indices["test"])

        print(f"Loaded dataset '{name}' with {len(dataset)} graphs.")
        print(dataset[0])


        
    def _get_dataloader(self, dataset: Subset, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, **kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._get_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._get_dataloader(self.val_ds)

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return self._get_dataloader(self.test_ds)

    def train_inference_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._get_dataloader(self.train_ds, shuffle=False)

class OpenQDCToPyG(InMemoryDataset):
    def __init__(self, oqdc_ds, root="data/datasets/",
                 x_mode="one_hot", y_key="formation_energies",
                 y_index=0, max_entries=100000,
                 standardize_y: bool = True,
                 cutoff: float = None,
                 transform=None, pre_transform=None,
                 seed = 42):
        self.oqdc_ds = oqdc_ds
        self.root = root
        self.x_mode = x_mode
        self.y_key = y_key
        self.y_index = y_index
        self.max_entries = max_entries
        self.standardize_y = standardize_y
        self.cutoff = cutoff

        self.element_list = sorted(int(z) for z in oqdc_ds.numbers)
        self.element2idx = {Z: i for i, Z in enumerate(self.element_list)}
        self.atom_types = len(self.element_list)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"data_{self.x_mode}_{self.y_key}_{self.y_index}_{self.max_entries}_{int(self.standardize_y)}.pt"]

    @property
    def raw_file_names(self): 
        return []

    def _x_from_mode(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        if self.x_mode == "Z":
            return z.unsqueeze(-1).to(torch.float)
        if self.x_mode == "none":
            return None
        if self.x_mode == "one_hot":
            idx = torch.tensor(
                [self.element2idx[int(zi)] for zi in z.tolist()],
                dtype=torch.long
            )
            return F.one_hot(idx, num_classes=self.atom_types).to(torch.float)
        raise ValueError(f"Unknown x_mode={self.x_mode}")

    def process(self):
        data_list = []
        y_values = []
        n_atoms_list = []

        i = 0
        for entry in tqdm(self.oqdc_ds, desc="OpenQDC → PyG"):
            pos = entry["positions"].to(torch.float)
            z = entry["atomic_numbers"].to(torch.long)

            y = entry[self.y_key][self.y_index].reshape(1, 1).to(torch.float)
            y_values.append(y.item())
            n_atoms_list.append(len(z))   
            x = self._x_from_mode(z)

            d = Data(
                x=x, pos=pos, z=z, y=y.clone(), 
                name=str(entry.get("name", "")), subset=str(entry.get("subset", ""))
            )
            if self.pre_transform:
                d = self.pre_transform(d)
            data_list.append(d)

            i += 1
            if self.max_entries and i >= self.max_entries:
                break

        if self.standardize_y:
            y_values = torch.tensor(y_values, dtype=torch.float)
            n_atoms_list = torch.tensor(n_atoms_list, dtype=torch.float)

            per_atom_residuals = y_values / n_atoms_list
            sigma_res = per_atom_residuals.std()
            mean_res = per_atom_residuals.mean()

            for j, d in enumerate(data_list):
                d.y = d.y / sigma_res

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        if self.standardize_y:
            data.std_res = sigma_res
            data.mean_res = mean_res 
        data.atom_types = self.atom_types
        data.group_attr = 'name'
        torch.save((data, slices), self.processed_paths[0])

class GraphJEPASampler(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        context_ratio: float = 0.6,
        target_ratio: float = 0.1,
    ):
        """
        Wraps a graph-level dataset with JEPA-style context/target sampling.

        Args:
            base_dataset: dataset of Data objects
            context_ratio: fraction of nodes to use as context
            target_ratio: fraction of remaining nodes to use as target
        """
        self.base_dataset = base_dataset
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        data: Data = self.base_dataset[idx]

        num_nodes = data.num_nodes

        assert num_nodes > 1

        all_nodes = list(range(num_nodes))
        random.shuffle(all_nodes)

        # context nodes
        num_context = int(self.context_ratio * num_nodes)

        context_nodes = all_nodes[:num_context]

        # target nodes (from remaining)
        remaining_nodes = all_nodes[num_context:]
        n_remaining = len(remaining_nodes)
        num_target = int(self.target_ratio * n_remaining)
        num_target = max(1, num_target)
        target_nodes = remaining_nodes[:num_target]

        # build masks
        context_mask = torch.zeros(num_nodes, dtype=torch.bool)
        context_mask[context_nodes] = True
        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_mask[target_nodes] = True

        # filter edges
        edge_index = data.edge_index
        context_edges = edge_index[:, context_mask[edge_index[0]] & context_mask[edge_index[1]]]

        if hasattr(data, "positional_encoding"):
            pos_enc = data.positional_encoding
            context_pos = pos_enc[context_mask]
        else:
            pos_enc = None
            context_pos = None

        if hasattr(data, "z"):
            z = data.z
            context_z = z[context_mask]
        else:
            z = None
            context_z = None

        if hasattr(data, "pos"):
            position = data.pos
            context_position = position[context_mask]
        x = data.x

        context_x = x[context_mask]

        y = data.y

        # pack into Data objects
        return {
            "original": data,
            "context": Data(
                x=context_x,
                edge_index=self._remap_edges(context_edges, context_nodes),
                y=y,
                positional_encoding=context_pos,
                z=context_z,
                pos=context_position
                
            ),
            "targets": [
                Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    positional_encoding=pos_enc,
                    z=z,
                    target_mask=target_mask,
                    pos=position
                    
                )
            ],
        }
        
    def _remap_edges(self, edge_index, kept_nodes: list[int]):
        """Reindex edges so node indices are contiguous after subsetting."""
        if edge_index.numel() == 0:
            return edge_index

        # create a mapping from old node idx to new idx for only kept nodes
        old_to_new = {old: new for new, old in enumerate(sorted(kept_nodes))}

        # apply mapping to edge_index
        remapped_edges = edge_index.clone()

        for i, node_idx in enumerate(remapped_edges):
            remapped_edges[i] = node_idx.apply_(lambda x: old_to_new[x])

        return remapped_edges

class GraphLevelJEPASamplingDataModule(GraphLevelDataModule):
    def __init__(
        self,
        dataset_config: GraphLevelDatasetConfig,
        batch_size: int,
        pos_enc_path: Path = None,
        context_ratio: float = 0.6,
        target_ratio: float = 0.1,
    ):
        super().__init__(dataset_config, batch_size, pos_enc_path)
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None

        wrapped_train = GraphJEPASampler(
            self.train_ds,
            context_ratio=self.context_ratio,
            target_ratio=self.target_ratio,
        )
        return DataLoader(
            wrapped_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

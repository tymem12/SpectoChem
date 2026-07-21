from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split, Subset
import os
from typing import Optional, Any
from collections.abc import Sequence
from sklearn.model_selection import train_test_split

import random

from sklearn.model_selection import GroupShuffleSplit

import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Dataset, InMemoryDataset
from torch_geometric.datasets import ZINC, QM9, TUDataset
from torch_geometric.transforms import BaseTransform

from gjepa.utils.pyg import create_transform

from gjepa.config import GraphLevelDatasetConfig
from gjepa.datasets.node_level import GraphDataModule
from gjepa.utils.pos_encoding import attach_pe_to_dataset_inplace
from gjepa.utils.graph_level import split_dataset
import torch.nn.functional as F
from gjepa.datasets.cv_vis.tmqmg_star import TMQMGStarDataset
from gjepa.datasets.cv_vis.standarizer_singleton import StandarizerSingletonF, StandarizerSingletonLambda

def load_graph(
    root_dir: Path,
    name: str,
    transform: BaseTransform | None = None,
    pre_transform: BaseTransform | None = None,
    additional_loading_params: dict[str, Any] | None = None
) -> Dataset:
    print(f"Loading dataset '{name}' from {root_dir}...")
    print(additional_loading_params)
    kwargs = dict(root=root_dir, transform=transform, pre_transform=pre_transform) | additional_loading_params

    if name == "TMQM_SPECTO":
        dataset = TMQMGStarDataset(**kwargs)
    else:
        raise ValueError(f"Invalid dataset name in config: {name!r}")

    return dataset

class GraphLevelDataModule(GraphDataModule):
    def __init__(
        self,
        dataset_config: GraphLevelDatasetConfig,
        batch_size: int,
        random_seed: Optional[int] = None,
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

        if random_seed is None:
            random_seed = torch.initial_seed()

        self.random_seed = random_seed
        self.generator = torch.Generator()

    def reseed_generator(self) -> torch.Generator:
        self.generator.manual_seed(self.random_seed)
        return self.generator

    def setup(self, stage: str) -> None:
        # torch trainer calls `setup()` again with `stage="fit` during testing, which causes setting
        # train, val, test ds again - with a different split - which means a very possible data leak
        if stage != "fit":
            return

        name = self.config.name

        split_ratios = self.config.split_ratios

        should_split = name != ZINC.__name__

        block_3_split_mode = self.config.block_3_split_mode
        is_binary_task = self.config.task_type == "binary"

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
        else:
            if split_ratios is not None:
                raise ValueError(
                    f"{name!r} dataset uses official splits; `split_ratios` must be `None`."
                )

            assert block_3_split_mode is None, f"{name!r} uses official splits; `block_3_split_mode` must be None."
            assert not self.config.group_by_isomers, f"{name!r} uses official splits; `group_by_isomers` must be False."

        if block_3_split_mode is not None:
            if self.config.additional_loading_params is None:
                self.config.additional_loading_params = {}

            additional_loading_params = self.config.additional_loading_params

            old_b3 = additional_loading_params.get('block_3_only', 'missing')
            old_mark = additional_loading_params.get('mark_block_3', 'missing')

            print(f"Notice: `config.block_3_split_mode` is {block_3_split_mode!r}. "
                  f"Overwriting additional_loading_params: `block_3_only` ({old_b3!r} -> False), "
                  f"`mark_block_3` ({old_mark!r} -> True).")

            additional_loading_params['block_3_only'] = False
            additional_loading_params['mark_block_3'] = True

        dataset = load_graph(
            root_dir=self.config.root_dir,
            name=name,
            additional_loading_params=self.config.additional_loading_params,
            pre_transform=create_transform(self.config.pre_transforms),
            transform=create_transform(self.config.transforms),
        )

        if self.pos_enc_path is not None:
            attach_pe_to_dataset_inplace(dataset=dataset, pe_path=self.pos_enc_path)

        def _get_isomer_key(d):
            return str(sorted(d.z.tolist()))

        def _get_binary_label(d):
            # Extracts scalar label, assumes class 1 is the positive representation
            return int(d.y.view(-1)[0].item())

        def _isomer_group_split(ds, train_size, stratify=False):
            groups = list(map(_get_isomer_key, ds))
            
            if stratify:
                labels = list(map(_get_binary_label, ds))
                # Aggregate labels to the group level. If any isomer has class 1, the group gets class 1
                group_to_label = {}
                for g, l in zip(groups, labels):
                    group_to_label[g] = max(group_to_label.get(g, 0), l)
                
                unique_groups = list(group_to_label.keys())
                group_labels = [group_to_label[g] for g in unique_groups]
                
                # Stratify at the group level to prevent data leakage between splits
                train_groups, test_groups = train_test_split(
                    unique_groups, train_size=train_size, stratify=group_labels, random_state=self.random_seed
                )
                
                train_g_set = set(train_groups)
                idx1 = [i for i, g in enumerate(groups) if g in train_g_set]
                idx2 = [i for i, g in enumerate(groups) if g not in train_g_set]
                
                return Subset(ds, idx1), Subset(ds, idx2)
            else:
                gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=self.random_seed)
                idx1, idx2 = next(gss.split(range(len(ds)), groups=groups))
                return Subset(ds, idx1), Subset(ds, idx2)

        def get_isomer_sets(ds):
            return set(map(_get_isomer_key, ds))

        if should_split:
            if block_3_split_mode is None:
                if self.config.group_by_isomers:
                    train_r, val_r = split_ratios[0], split_ratios[1]
                    train_val_ds, self.test_ds = _isomer_group_split(
                        dataset, train_r + val_r, stratify=is_binary_task
                    )

                    norm_train_r = train_r / (train_r + val_r)
                    self.train_ds, self.val_ds = _isomer_group_split(
                        train_val_ds, norm_train_r, stratify=is_binary_task
                    )
                else:
                    if is_binary_task:
                        train_r, val_r = split_ratios[0], split_ratios[1]
                        test_r = 1.0 - train_r - val_r
                        labels = list(map(_get_binary_label, dataset))
                        indices = list(range(len(dataset)))
                        
                        # Split into train+val and test
                        tv_idx, test_idx, tv_labels, _ = train_test_split(
                            indices, labels, test_size=test_r, stratify=labels, random_state=self.random_seed
                        )
                        # Split train+val into train and val
                        norm_train_r = train_r / (train_r + val_r)
                        train_idx, val_idx = train_test_split(
                            tv_idx, train_size=norm_train_r, stratify=tv_labels, random_state=self.random_seed
                        )
                        
                        self.train_ds = Subset(dataset, train_idx)
                        self.val_ds = Subset(dataset, val_idx)
                        self.test_ds = Subset(dataset, test_idx)
                    else:
                        self.train_ds, self.val_ds, self.test_ds = split_dataset(
                            dataset, split_ratios, self.reseed_generator()
                        )
            else:
                block_3_indices = []
                non_block_3_indices = []

                for i, d in enumerate(dataset):
                    if d.is_from_block_3:
                        container = block_3_indices
                    else:
                        container = non_block_3_indices

                    container.append(i)

                block_3_subset = Subset(dataset, block_3_indices)
                non_block_3_subset = Subset(dataset, non_block_3_indices)

                train_r, val_r = split_ratios[0], split_ratios[1]
                norm_train_r = train_r / (train_r + val_r)

                match block_3_split_mode:
                    case "train":
                        train_val_pool, test_pool = block_3_subset, non_block_3_subset
                    case "test":
                        train_val_pool, test_pool = non_block_3_subset, block_3_subset
                    case _:
                        raise ValueError(f"Invalid `block_3_split_mode` {block_3_split_mode}")

                self.test_ds = test_pool

                if self.config.group_by_isomers:
                    self.train_ds, self.val_ds = _isomer_group_split(
                        train_val_pool, norm_train_r, stratify=is_binary_task
                    )
                else:
                    if is_binary_task:
                        labels = list(map(_get_binary_label, train_val_pool))
                        indices = list(range(len(train_val_pool)))
                        train_idx, val_idx = train_test_split(
                            indices, train_size=norm_train_r, stratify=labels, random_state=self.random_seed
                        )
                        self.train_ds = Subset(train_val_pool, train_idx)
                        self.val_ds = Subset(train_val_pool, val_idx)
                    else:
                        num_train = int(len(train_val_pool) * norm_train_r)
                        num_val = len(train_val_pool) - num_train
                        self.train_ds, self.val_ds = random_split(train_val_pool, [num_train, num_val], self.reseed_generator())
        else:
            self.train_ds = Subset(dataset, dataset.split_indices["train"])
            self.val_ds   = Subset(dataset, dataset.split_indices["val"])
            self.test_ds  = Subset(dataset, dataset.split_indices["test"])

        if self.config.group_by_isomers:
            train_isomers = get_isomer_sets(self.train_ds)
            val_isomers = get_isomer_sets(self.val_ds)
            test_isomers = get_isomer_sets(self.test_ds)

            assert train_isomers.isdisjoint(val_isomers), "Data leakage: Train & Val share isomers"
            assert train_isomers.isdisjoint(test_isomers), "Data leakage: Train & Test share isomers"
            assert val_isomers.isdisjoint(test_isomers), "Data leakage: Val & Test share isomers"

            print(f"Unique isomers: train: {len(train_isomers)} | val: {len(val_isomers)} | test: {len(test_isomers)}")

        print(self.test_ds[0].y)

        self._standarize_output(output_type=self.config.additional_loading_params['prediction_type'],
                                standarize_lambda=self.config.additional_loading_params['standarize_lambda'],
                                standarize_f=self.config.additional_loading_params['standarize_f'])
        
        full_ds_len = sum(map(len, (
            self.train_ds, self.val_ds, self.test_ds
        )))

        print("Total dataset len:", full_ds_len)

        if is_binary_task:
            def print_split_stats(split_name, ds):
                if ds is None or len(ds) == 0:
                    print(f"len of {split_name} is 0")
                    return
                
                # Extract all labels for this split
                labels = [_get_binary_label(d) for d in ds]
                total = len(labels)
                pos_count = sum(labels)
                neg_count = total - pos_count
                pos_pct = (pos_count / total) * 100
                
                print(f"len of {split_name} is {total} ({total/full_ds_len:.2%} of all) | Pos: {pos_count} ({pos_pct:.2f}%) | Neg: {neg_count}")

            print_split_stats("train", self.train_ds)
            print_split_stats("val", self.val_ds)
            print_split_stats("test", self.test_ds)
        else:
            for ds_label, ds_subset in(
                ("train", self.train_ds),
                ("val", self.val_ds),
                ("test", self.test_ds)
            ):
                ds_len = len(ds_subset)

                print(f"len of {ds_label} is {ds_len} ({ds_len/full_ds_len:.2%} of all)")

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

    def _standarize_output(self, output_type: str, standarize_lambda: bool, standarize_f: bool):
        if self.train_ds is None:
            raise RuntimeError("train_ds is not initialized. Call setup() before _standarize_output().")

        if output_type == "multi_regressor":
            if not standarize_lambda and not standarize_f:
                return  # no-op if both are False

            lambda_vals = []
            f_vals = []

            # 1. Gather stats ONLY from the train_ds to prevent data leakage
            for data_element in self.train_ds:
                y = data_element.y.view(-1)
                num_states = y.size(0) // 2  # First half is lambda, second half is f
                
                if standarize_lambda:
                    lambda_vals.extend(y[:num_states].tolist())
                if standarize_f:
                    f_vals.extend(y[num_states:].tolist())

            lambda_mean = lambda_std = None
            f_mean = f_std = None

            # 2. Compute Mean/Std and register with singletons
            if standarize_lambda:
                if len(lambda_vals) == 0:
                    raise RuntimeError("No lambda values found in train set.")
                lambda_tensor = torch.tensor(lambda_vals, dtype=torch.float32)
                lambda_mean = lambda_tensor.mean()
                lambda_std  = lambda_tensor.std(unbiased=False)
                StandarizerSingletonLambda.set_values(mean_lambda=lambda_mean, std_lambda=lambda_std)

            if standarize_f:
                if len(f_vals) == 0:
                    raise RuntimeError("No f values found in train set.")
                f_tensor = torch.tensor(f_vals, dtype=torch.float32)
                f_mean = f_tensor.mean()
                f_std  = f_tensor.std(unbiased=False)
                StandarizerSingletonF.set_values(mean_f=f_mean, std_f=f_std)

            self._y_mean = torch.tensor([
                lambda_mean if lambda_mean is not None else 0.0,
                f_mean if f_mean is not None else 0.0
            ])
            self._y_std = torch.tensor([
                lambda_std if lambda_std is not None else 1.0,
                f_std if f_std is not None else 1.0
            ])

            # 3. Apply the train-derived statistics to all subsets
            def _standardize_dataset(ds):
                if ds is None:
                    return None
                standardized = []
                for i in range(len(ds)):
                    data = ds[i].clone()
                    y = data.y.view(-1).clone()
                    num_states = y.size(0) // 2
                    
                    if standarize_lambda:
                        y[:num_states] = (y[:num_states] - lambda_mean) / (lambda_std + 1e-8)
                    if standarize_f:
                        y[num_states:] = (y[num_states:] - f_mean) / (f_std + 1e-8)
                        
                    data.y = y.view_as(data.y)
                    standardized.append(data)
                return standardized

            self.train_ds = _standardize_dataset(self.train_ds)
            self.val_ds   = _standardize_dataset(self.val_ds)
            self.test_ds  = _standardize_dataset(self.test_ds)
            
            return

        if output_type == "pairs":
            if not standarize_lambda and not standarize_f:
                return  # no-op if both False

            lambda_vals = []
            f_vals = []

            for data_element in self.train_ds:
                y = data_element.y.view(-1)

                for idx, val in enumerate(y):
                    if idx % 2 == 0:
                        if standarize_lambda:
                            lambda_vals.append(val.item())
                    else:
                        if standarize_f:
                            f_vals.append(val.item())

            lambda_mean = lambda_std = None
            f_mean = f_std = None

            if standarize_lambda:
                if len(lambda_vals) == 0:
                    raise RuntimeError("No lambda values found in train set.")
                lambda_tensor = torch.tensor(lambda_vals, dtype=torch.float32)
                lambda_mean = lambda_tensor.mean()
                lambda_std  = lambda_tensor.std(unbiased=False)
                StandarizerSingletonLambda.set_values(mean_lambda=lambda_mean, std_lambda=lambda_std)

            if standarize_f:
                if len(f_vals) == 0:
                    raise RuntimeError("No f values found in train set.")
                f_tensor = torch.tensor(f_vals, dtype=torch.float32)
                f_mean = f_tensor.mean()
                f_std  = f_tensor.std(unbiased=False)
                StandarizerSingletonF.set_values(mean_f=f_mean, std_f=f_std)


            self._y_mean = torch.tensor([
                lambda_mean if lambda_mean is not None else 0.0,
                f_mean if f_mean is not None else 0.0
            ])
            self._y_std = torch.tensor([
                lambda_std if lambda_std is not None else 1.0,
                f_std if f_std is not None else 1.0
            ])

            def _standardize_dataset(ds):
                if ds is None:
                    return None
                standardized = []
                for i in range(len(ds)):
                    data = ds[i].clone()
                    y = data.y.view(-1).clone()
                    for idx in range(y.size(0)):
                        if idx % 2 == 0:
                            if standarize_lambda:
                                y[idx] = (y[idx] - lambda_mean) / (lambda_std + 1e-8)
                        else:
                            # f
                            if standarize_f:
                                y[idx] = (y[idx] - f_mean) / (f_std + 1e-8)
                    data.y = y.view_as(data.y)
                    standardized.append(data)
                return standardized

            self.train_ds = _standardize_dataset(self.train_ds)
            self.val_ds   = _standardize_dataset(self.val_ds)
            self.test_ds  = _standardize_dataset(self.test_ds)
            return

        if output_type == "vector":
            if not standarize_f:
                return
            f_vals = []

            for data_element in self.train_ds:
                y = data_element.y.view(-1)
                for val in y:
                    if val.item() != 0:
                        f_vals.append(val.item())

            if len(f_vals) == 0:
                raise RuntimeError("No non-zero f values found in train set for 'vector' standardization.")

            f_tensor = torch.tensor(f_vals, dtype=torch.float32)
            f_mean = f_tensor.mean()
            f_std = f_tensor.std(unbiased=False)
            StandarizerSingletonF.set_values(mean_f=f_mean, std_f=f_std)

            self._y_mean = f_mean
            self._y_std = f_std

            def _standardize_dataset(ds):
                if ds is None:
                    return None
                standardized = []
                for i in range(len(ds)):
                    data = ds[i].clone()
                    y = data.y.view(-1).clone()

                    for idx in range(y.size(0)):
                        # only standardize non-zero values
                        if y[idx].item() != 0:
                            y[idx] = (y[idx] - f_mean) / (f_std + 1e-8)

                    data.y = y.view_as(data.y)
                    standardized.append(data)
                return standardized

            self.train_ds = _standardize_dataset(self.train_ds)
            self.val_ds = _standardize_dataset(self.val_ds)
            self.test_ds = _standardize_dataset(self.test_ds)
            return

        if output_type == "only_lambdas":
            if not standarize_lambda:
                return

            lambda_vals = []

            for data_element in self.train_ds:
                y = data_element.y.view(-1)
                for val in y:
                    lambda_vals.append(val.item())

            if len(lambda_vals) == 0:
                raise RuntimeError("No lambda values found in train set for 'only_lambdas' standardization.")

            lambda_tensor = torch.tensor(lambda_vals, dtype=torch.float32)
            lambda_mean = lambda_tensor.mean()
            lambda_std = lambda_tensor.std(unbiased=False)
            StandarizerSingletonLambda.set_values(mean_lambda=lambda_mean, std_lambda=lambda_std)

            self._y_mean = lambda_mean
            self._y_std = lambda_std

            def _standardize_dataset(ds):
                if ds is None:
                    return None
                standardized = []
                for i in range(len(ds)):
                    data = ds[i].clone()
                    y = data.y.view(-1).clone()

                    for idx in range(y.size(0)):
                        y[idx] = (y[idx] - lambda_mean) / (lambda_std + 1e-8)

                    data.y = y.view_as(data.y)
                    standardized.append(data)
                return standardized

            self.train_ds = _standardize_dataset(self.train_ds)
            self.val_ds = _standardize_dataset(self.val_ds)
            self.test_ds = _standardize_dataset(self.test_ds)
            return
        if output_type == "f_regressor" or output_type == "lambda_regressor":
            
            if not standarize_f and not standarize_lambda:
                print("No standardization applied since both standarize_f and standarize_lambda are False.")
                return
            if not standarize_f and output_type == "f_regressor":
                print("No standardization applied to f values since standarize_f is False.")
                return
            if not standarize_lambda and output_type == "lambda_regressor":
                print("No standardization applied to lambda values since standarize_lambda is False.")
                return

            vals = []

            for data_element in self.train_ds:
                y = data_element.y.view(-1)
                for val in y:
                    vals.append(val.item())

            if len(vals) == 0:
                raise RuntimeError("No f values found in train set for 'f_regressor' standardization.")

            values_tensor = torch.tensor(vals, dtype=torch.float32)
            val_mean = values_tensor.mean()
            val_std = values_tensor.std(unbiased=False)

            def _standardize_dataset(ds):
                if ds is None:
                    return None
                standardized = []
                for i in range(len(ds)):
                    data = ds[i].clone()
                    y = data.y.view(-1).clone()
                    for idx in range(y.size(0)):
                        y[idx] = (y[idx] - val_mean) / (val_std + 1e-8)

                    data.y = y.view_as(data.y)
                    standardized.append(data)
                return standardized
            self.train_ds = _standardize_dataset(self.train_ds)
            self.val_ds = _standardize_dataset(self.val_ds)
            self.test_ds = _standardize_dataset(self.test_ds)
            if output_type == "f_regressor" and standarize_f:
                StandarizerSingletonF.set_values(mean_f=val_mean, std_f=val_std)
            elif output_type == "lambda_regressor" and standarize_lambda:
                StandarizerSingletonLambda.set_values(mean_lambda=val_mean, std_lambda=val_std)
            return
        if output_type in ["binary_classification",'binary_vector_multiclass', 'binary_vector_multilabel']:
            print('SHAPE: ', self.train_ds[0].y.shape)
            print('VALUES: ', self.train_ds[0].y)
            return
        else:
            raise ValueError(f"Unknown standarization type: {type!r}. Expected 'pairs', 'vector', or 'only_lambdas'.")

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

        num_context = int(self.context_ratio * num_nodes)

        context_nodes = all_nodes[:num_context]

        remaining_nodes = all_nodes[num_context:]
        n_remaining = len(remaining_nodes)
        num_target = int(self.target_ratio * n_remaining)
        num_target = max(1, num_target)
        target_nodes = remaining_nodes[:num_target]

        context_mask = torch.zeros(num_nodes, dtype=torch.bool)
        context_mask[context_nodes] = True
        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_mask[target_nodes] = True

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

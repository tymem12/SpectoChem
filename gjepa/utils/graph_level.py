from typing import Optional

import torch
import random

from torch.utils.data import Subset, random_split

def calc_edge_weight(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Calculate edge weights as Euclidean distances between connected nodes.

    Parameters
    ----------
    pos : torch.Tensor
        Shape [num_nodes, 3], atom positions in Ångströms.
    edge_index : torch.Tensor
        Shape [2, num_edges], COO-format edge index.

    Returns
    -------
    edge_weight : torch.Tensor
        Shape [num_edges], distances for each edge.
    """
    row, col = edge_index
    diff = pos[row] - pos[col]             # [E, 3]
    edge_weight = diff.norm(dim=-1).clamp(min=1e-12)        # [E]
    return edge_weight


def split_dataset(dataset, split_ratios, seed: Optional[int | torch.Generator] = 2137):

    test_ratio = round(1 - sum(split_ratios), 10)

    if getattr(dataset, "group_attr", None) is not None:
        assert isinstance(seed, int)

        group_attr = dataset.group_attr
        rng = random.Random(seed)

        groups = {}
        for idx, data in enumerate(dataset):
            if not hasattr(data, group_attr):
                raise AttributeError(
                    f"Element dataset[{idx}] nie ma atrybutu '{group_attr}'."
                )
            gval = getattr(data, group_attr)
            groups.setdefault(gval, []).append(idx)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        r_train, r_val = split_ratios
        n_groups = len(group_keys)
        n_train = int(r_train * n_groups)
        n_val   = int(r_val   * n_groups)
        n_test  = n_groups - n_train - n_val

        train_groups = group_keys[:n_train]
        val_groups   = group_keys[n_train:n_train + n_val]
        test_groups  = group_keys[n_train + n_val:]

        train_idx = [i for g in train_groups for i in groups[g]]
        val_idx   = [i for g in val_groups   for i in groups[g]]
        test_idx  = [i for g in test_groups  for i in groups[g]]

        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx),
        )

    else:
        assert seed is None or isinstance(seed, torch.Generator)

        return random_split(
            dataset,
            [*split_ratios, test_ratio],
            seed
        )

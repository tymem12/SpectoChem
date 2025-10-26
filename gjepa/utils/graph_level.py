from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_cluster import radius_graph
import torch
from torch.utils.data import Subset
import random

from torch.utils.data import Subset, random_split
import random
import torch

def edge_index_from_smiles(smiles: str, atomic_numbers: torch.Tensor, add_hs: bool = False):
    """
    Build edge_index (and edge_attr) from SMILES, ensuring atom order matches atomic_numbers.
    - atomic_numbers: 1D Long/Int tensor like entry["atomic_numbers"] (includes H if present).
    - add_hs: set True only if your atomic_numbers INCLUDE explicit H but SMILES is missing them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_hs:
        mol = Chem.AddHs(mol)

    rdkit_Z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    print(f"SMILES: {smiles}, RDKit atom order: {rdkit_Z.tolist()}")
    # Hard check: RDKit atom order must match your tensor order
    if rdkit_Z.dtype != atomic_numbers.dtype:
        atomic_numbers = atomic_numbers.to(rdkit_Z.dtype)

    if rdkit_Z.numel() != atomic_numbers.numel():
        raise ValueError(
            f"Atom count mismatch: SMILES has {rdkit_Z.numel()} atoms, "
            f"but atomic_numbers has {atomic_numbers.numel()}."
        )
    if not torch.equal(rdkit_Z, atomic_numbers):
        # same multiset?
        if torch.equal(torch.sort(rdkit_Z).values, torch.sort(atomic_numbers).values):
            raise ValueError(
                "SMILES atom order differs from entry['atomic_numbers'] order.\n"
                "Avoid canonicalizing SMILES and ensure you pass the original SMILES that matches "
                "the dataset’s atom ordering. If needed, we can add a renumbering step."
            )
        else:
            raise ValueError("SMILES composition does not match atomic_numbers.")

    rows, cols, attrs = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # undirected → two directed edges
        rows += [i, j]
        cols += [j, i]

        bt = b.GetBondType()
        feat = [
            int(bt == rdchem.BondType.SINGLE),
            int(bt == rdchem.BondType.DOUBLE),
            int(bt == rdchem.BondType.TRIPLE),
            int(bt == rdchem.BondType.AROMATIC),
            int(b.GetIsConjugated()),
            int(b.IsInRing()),
        ]
        attrs += [feat, feat]  # same attrs for both directions

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.long)  # or float if you prefer
    return edge_index, edge_attr




def calc_edge_index(pos: torch.Tensor, cutoff: float = 5.0) -> torch.Tensor:
    """
    Build edges by connecting atoms within a given cutoff radius.

    Parameters
    ----------
    pos : torch.Tensor
        Shape [num_nodes, 3], atom positions in Ångströms.
    cutoff : float
        Cutoff distance in Ångströms for edge construction.

    Returns
    -------
    edge_index : torch.Tensor
        Shape [2, num_edges], COO-format edge index.
    """
    # radius_graph builds edges for all pairs with distance <= cutoff
    # It does not create self-loops when loop=False
    edge_index = radius_graph(pos, r=cutoff, loop=False, max_num_neighbors=64)
    return edge_index

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


def split_dataset(dataset, split_ratios, seed=2137):

    test_ratio = round(1 - sum(split_ratios), 10)

    if getattr(dataset, "group_attr", None) is not None:
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
        return random_split(
            dataset,
            [*split_ratios, test_ratio]
        )

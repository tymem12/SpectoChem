from __future__ import annotations
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Sequence, Callable
from itertools import islice

class TMQMGStarDataset(Dataset):
    # Static mapping of periodic table blocks
    BLOCKS = {
        's': {'H', 'He', 'Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr', 'Cs', 'Ba', 'Fr', 'Ra'},
        'p': {'B', 'C', 'N', 'O', 'F', 'Ne', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
              'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
              'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'},
        'd': {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
              'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
              'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
              'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'},
        'f': {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
              'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'}
    }

    def __init__(
        self,
        data_root_path: str = "data/tmqmg",
        tmqmg_star_ds_file_name: str = "tmqmg_star.csv",
        tmqmg_xyz_dir_name: str = "xyz",
        feature_cols: Optional[Sequence[str]] = None,
        target_cols: Optional[Sequence[str]] = None,
        atom_filter: Optional[int | str | Sequence[str]] = None,
        transform: Optional[Callable] = None
    ) -> None:
        self.transform = transform

        csv_path = Path(data_root_path) / tmqmg_star_ds_file_name
        self.data = pd.read_csv(csv_path)
        xyz_dir = Path(data_root_path) / tmqmg_xyz_dir_name
        self.xyz_dir = xyz_dir

        # Determine atoms to filter
        if atom_filter is None:
            atoms_filter: Optional[set[str]] = None
        elif isinstance(atom_filter, int):
            # Select the block key by index using islice
            try:
                key = next(islice(self.BLOCKS.keys(), atom_filter - 1, atom_filter))
            except StopIteration:
                raise ValueError(f"Invalid block index {atom_filter}, must be 1..{len(self.BLOCKS)}")
            atoms_filter = self.BLOCKS[key]
        elif isinstance(atom_filter, str):
            atoms_filter = self.BLOCKS.get(atom_filter.lower())
            if atoms_filter is None:
                raise ValueError(f"Invalid block key '{atom_filter}', must be one of {list(self.BLOCKS.keys())}")
        else:
            atoms_filter = set(atom_filter)  # assume sequence of element symbols

        # Filter molecules
        if atoms_filter is not None:
            filtered_ids = []
            for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Filtering molecules"):
                xyz_file = xyz_dir / f"{row['id']}.xyz"
                if not xyz_file.exists():
                    continue
                with open(xyz_file, "r") as f:
                    lines = f.readlines()[2:]  # skip first two lines
                    atoms_in_mol = [line.split()[0] for line in lines]
                    if atoms_filter.intersection(atoms_in_mol):
                        filtered_ids.append(row['id'])
            self.data = self.data[self.data['id'].isin(filtered_ids)].reset_index(drop=True)

        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in self.data.select_dtypes(include='number').columns
                            if c not in (target_cols or []) and c != 'id']
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.X = self.data[self.feature_cols].to_numpy()
        self.y = self.data[self.target_cols].to_numpy() if self.target_cols else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        X = torch.tensor(self.X[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.float) if self.y is not None else None
        if self.transform:
            X = self.transform(X)
        return X, y

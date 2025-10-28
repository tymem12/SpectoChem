# custom_chemical_dataset.py
from __future__ import annotations
import os
import json
import ast
import hashlib
from typing import List, Optional, Sequence, Union, Any, Dict

import torch
from torch import Tensor
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from gjepa.utils.cv_vis import _SYMBOL2Z, _is_number, _to_list, _parse_coords, _parse_atom_types, _to_float_vec

from tqdm import tqdm


class SpectoDataset(InMemoryDataset):
    SUBSET_TO_FILE = {
        "example": "raw/tmqm_EXAMPLE.csv",
        "cv_vis": "raw/tmqm_uvvis_final_40k.csv",   
        "all": "raw/tmQM_Xall.csv",                 
    }

    def __init__(
        self,
        root: str,
        subset: str,
        y_columns: Optional[Sequence[str]] = None,
        extra_fields: Optional[Sequence[str]] = None,
        transform=None,
        pre_transform=None,
        force_reprocess: bool = False,
        version: str = "v1",
    ):
        self.subset = subset
        self.y_columns = list(y_columns) if y_columns is not None else []
        self.extra_fields = list(extra_fields) if extra_fields is not None else []
        self.version = str(version)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        if force_reprocess:
            try:
                os.remove(self.processed_paths[0])
            except FileNotFoundError:
                pass

        data, slices = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices = data, slices

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return [self._raw_csv_relpath()]

    @property
    def processed_file_names(self) -> Union[str, List[str]]:
        sig = {
            "subset": self.subset,
            "y_columns": tuple(self.y_columns),
            "extra_fields": tuple(self.extra_fields),
            "version": self.version,
            "pre_transform": repr(self.pre_transform.__class__.__name__) if self.pre_transform else "none",
        }
        h = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
        return [f"tmqm_{self.subset}_y{len(self.y_columns)}_{h}.pt"]

    def download(self):
        pass

    def process(self):
        csv_path = os.path.join(self.root, self._raw_csv_relpath())
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found for subset '{self.subset}': {csv_path}")

        df = pd.read_csv(csv_path)

        required = ["atom_coords", "atom_types", "SMILES", "origin_ID", "CSD_code"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in CSV: {missing}")

        if any(c.upper() == "ABSORPTION_SPECTOGRAM" for c in self.y_columns):
            raise NotImplementedError(
                "ABSORPTION_SPECTOGRAM selected. NOW WE DO NOT HAVE THE DATA"
            )

        data_list: List[Data] = []
        y_dim: Optional[int] = len(self.y_columns) if self.y_columns else None

        for i, row in tqdm(df.iterrows()):
            try:
                num_atoms = int(row["num_atoms"]) if "num_atoms" in df.columns and not pd.isna(row["num_atoms"]) else None
                pos = _parse_coords(row["atom_coords"], expected_n=num_atoms)
                z = _parse_atom_types(row["atom_types"])
                if pos.size(0) != z.numel():
                    raise ValueError(f"Row {i}: pos has {pos.size(0)} atoms but z has {z.numel()}.")

                smiles = "" if pd.isna(row["SMILES"]) else str(row["SMILES"])
                origin_id = None if pd.isna(row["origin_ID"]) else str(row["origin_ID"])
                csd_code = None if pd.isna(row["CSD_code"]) else str(row["CSD_code"])

                kwargs = dict(pos=pos, z=z, smiles=smiles, origin_id=origin_id, CSD_code=csd_code)

                if self.y_columns:
                    vals = []
                    for col in self.y_columns:
                        if col not in df.columns:
                            raise KeyError(f"Requested y column '{col}' not found in CSV.")
                        vals.append(row[col])
                    y = torch.tensor([float(v) for v in vals], dtype=torch.float32).unsqueeze(0)  # -> shape (1, C)
                    if y_dim is not None and y.numel() != y_dim:
                        raise ValueError(f"Row {i}: y dim mismatch (got {y.numel()}, expected {y_dim}).")
                    kwargs["y"] = y

                for col in self.extra_fields:
                    if col not in df.columns:
                        raise KeyError(f"Requested extra field '{col}' not found in CSV.")
                    val = row[col]
                    parsed = None
                    if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
                        maybe = _to_list(val)
                        if all(_is_number(x) for x in maybe):
                            parsed = torch.tensor([float(x) for x in maybe], dtype=torch.float32)
                        else:
                            parsed = maybe
                    kwargs[col] = parsed if parsed is not None else val

                data = Data(**kwargs)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            except Exception as e:
                raise RuntimeError(f"Error parsing row {i}: {e}") from e

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])    # ---- helpers ----


    def _raw_csv_relpath(self) -> str:
        rel = self.SUBSET_TO_FILE.get(self.subset)
        if rel is None:
            raise ValueError(f"Unsupported subset '{self.subset}'.")
        return rel

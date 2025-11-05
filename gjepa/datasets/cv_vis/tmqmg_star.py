import os
import json
import hashlib
from typing import Optional, Sequence

import torch

import pandas as pd

from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data

from gjepa.utils.cv_vis import _is_number, _to_list, _parse_coords, _parse_atom_types

class TMQMGStarDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        block_3_only: bool = False,
        y_columns: Optional[Sequence[str]] = None,
        extra_fields: Optional[Sequence[str]] = None,
        transform=None,
        pre_transform=None,
        force_reprocess: bool = False,
        version: str = "v1",
    ):
        self.y_columns = list(y_columns) if y_columns else []
        self.extra_fields = list(extra_fields) if extra_fields else []
        self.block_3_only = block_3_only
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
    def raw_file_names(self) -> list[str]:
        return [self._csv_filename()]

    @property
    def processed_file_names(self) -> list[str]:
        sig = {
            "y_columns": tuple(self.y_columns),
            "extra_fields": tuple(self.extra_fields),
            "block_3_only": self.block_3_only,
            "version": self.version,
            "pre_transform": repr(self.pre_transform.__class__.__name__) if self.pre_transform else "none",
        }
        h = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
        return [f"tmqmg-star_block3-{self.block_3_only}_y{len(self.y_columns)}_{h}.pt"]

    def _csv_filename(self) -> str:
        """Select correct CSV depending on block_3_only flag."""
        if self.block_3_only:
            return "raw/uvvis_final_40k.csv"

        return "raw/tmqm_all.csv"

    def download(self):
        """Dataset assumed to be locally available."""
        pass

    def process(self):
        base_csv_path = os.path.join(self.root, self._csv_filename())
        tmqmg_star_path = os.path.join(self.root, "raw", "tmqmg_star.csv")

        if not os.path.exists(base_csv_path):
            raise FileNotFoundError(f"Base CSV not found: {base_csv_path}")
        if not os.path.exists(tmqmg_star_path):
            raise FileNotFoundError(f"tmqmg_star.csv not found: {tmqmg_star_path}")

        df_base = pd.read_csv(base_csv_path)
        df_star = pd.read_csv(tmqmg_star_path)

        if "CSD_code" not in df_base.columns:
            raise KeyError("Missing 'CSD_code' column in base CSV.")
        if "id" not in df_star.columns:
            raise KeyError("Missing 'id' column in tmqmg_star.csv.")

        overlap = set(df_base.columns) & set(df_star.columns) - {"CSD_code", "id"}
        if overlap:
            raise ValueError(f"Overlapping columns between base and star CSVs: {sorted(overlap)}")

        df = pd.merge(
            df_base,
            df_star,
            how="inner",
            left_on="CSD_code",
            right_on="id"
        )

        print(f"Merged dataset: {len(df)} rows (from {len(df_base)} base and {len(df_star)} star)")

        required = ["atom_coords", "atom_types", "SMILES", "origin_ID", "CSD_code"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in merged dataset: {missing}")

        if any(c.upper() == "ABSORPTION_SPECTOGRAM" for c in self.y_columns):
            raise NotImplementedError(
                "ABSORPTION_SPECTOGRAM selected. NOW WE DO NOT HAVE THE DATA"
            )

        data_list: list[Data] = []
        y_dim: Optional[int] = len(self.y_columns) if self.y_columns else None

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing TMQMG* merged"):
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
                            raise KeyError(f"Requested y column '{col}' not found in merged dataset.")
                        vals.append(row[col])
                    y = torch.tensor([float(v) for v in vals], dtype=torch.float32).unsqueeze(0)
                    if y_dim is not None and y.numel() != y_dim:
                        raise ValueError(f"Row {i}: y dim mismatch (got {y.numel()}, expected {y_dim}).")
                    kwargs["y"] = y

                for col in self.extra_fields:
                    if col not in df.columns:
                        raise KeyError(f"Requested extra field '{col}' not found in merged dataset.")
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
        torch.save((data, slices), self.processed_paths[0])

import os
import json
import hashlib
from typing import Optional, Sequence, Tuple, List
import torch
import pandas as pd
import numpy as np
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
        prediction_type: str = "pairs",
        prediction_params: Optional[dict] = None,
        vis_range: Tuple[float, float] = (400.0, 700.0),
        max_states: int = 10
    ):
        self.y_columns = list(y_columns) if y_columns else []
        self.extra_fields = list(extra_fields) if extra_fields else []
        self.block_3_only = block_3_only
        self.version = str(version)

        self.prediction_type = prediction_type
        self.prediction_params = prediction_params or {}
        self.min_lambda, self.max_lambda = vis_range
        self.max_states = max_states

        if self.prediction_type not in {"pairs", "vector"}:
            raise ValueError(f"Invalid prediction_type: {self.prediction_type}")

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
            "prediction_type": self.prediction_type,
            "prediction_params": self.prediction_params,
            "vis_range": (self.min_lambda, self.max_lambda),
        }
        h = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
        return [f"tmqmg-star_block3-{self.block_3_only}_y{len(self.y_columns)}_{h}.pt"]

    def _csv_filename(self) -> str:
        return "raw/uvvis_final_40k.csv" if self.block_3_only else "raw/tmqm_all.csv"

    def download(self):
        pass

    def _filter_visible_transitions(self, row: pd.Series) -> List[Tuple[float, float]]:
        """
        Return the first 10 (lambda, f) pairs that are **all** inside the visible range.
        If any of the first 10 is missing or outside the range → return [] (molecule is dropped).
        """
        transitions = []
        for i in range(1, self.max_states):
            lam_col = f"lambda_{i}_gasphase"
            f_col = f"f_{i}_gasphase"

            if lam_col not in row or f_col not in row:
                return []

            lam = row[lam_col]
            f = row[f_col]

            if pd.isna(lam) or pd.isna(f):
                return []

            lam, f = float(lam), float(f)

            if not (self.min_lambda <= lam <= self.max_lambda):
                return []
            transitions.append((lam, f))

        return transitions

    def _build_top_pairs(self, transitions: List[Tuple[float, float]], num_pairs: int = 10) -> torch.Tensor:
        """First-k absorptions (by input order) in visible range → flat vector [λ1, f1, λ2, f2, ...]."""
        if not transitions:
            return torch.zeros(1, num_pairs * 2, dtype=torch.float32)

        selected = transitions[:num_pairs]

        vec = []
        for lam, f in selected:
            vec.extend([lam, f])

        while len(vec) < num_pairs * 2:
            vec.extend([0.0, 0.0])

        return torch.tensor([vec], dtype=torch.float32)

    def _build_absorption_vector(
        self,
        transitions: List[Tuple[float, float]],
        wavelength_range: Tuple[float, float] = (300.0, 700.0),
        num_bins: int = 400
    ) -> torch.Tensor:
        """Histogram-style discrete spectrum."""
        start, end = wavelength_range
        hist = np.zeros(num_bins)

        for lam, f in transitions:
            if start <= lam <= end:
                idx = int((lam - start) / (end - start) * num_bins)
                idx = min(max(idx, 0), num_bins - 1)
                hist[idx] += f

        return torch.tensor(hist, dtype=torch.float32).unsqueeze(0)

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

        data_list: List[Data] = []

        num_pairs = self.prediction_params.get("num_pairs", 10)
        vector_range = self.prediction_params.get("wavelength_range", (300.0, 700.0))
        num_bins = self.prediction_params.get("num_bins", 400)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                num_atoms = int(row["num_atoms"]) if "num_atoms" in row and not pd.isna(row["num_atoms"]) else None
                pos = _parse_coords(row["atom_coords"], expected_n=num_atoms)
                z = _parse_atom_types(row["atom_types"])

                if pos.size(0) != z.numel():
                    raise ValueError(f"Atom count mismatch: pos={pos.size(0)}, z={z.numel()}")

                smiles = "" if pd.isna(row["SMILES"]) else str(row["SMILES"])
                origin_id = None if pd.isna(row["origin_ID"]) else str(row["origin_ID"])
                csd_code = None if pd.isna(row["CSD_code"]) else str(row["CSD_code"])

                kwargs = dict(pos=pos, z=z, smiles=smiles, origin_id=origin_id, CSD_code=csd_code)

                if self.prediction_type in {"pairs", "vector"}:
                    transitions = self._filter_visible_transitions(row)
                    if len(transitions) != self.max_states:
                        continue
                    if self.prediction_type == "pairs":
                        y = self._build_top_pairs(transitions, num_pairs=num_pairs)
                    else:
                        y = self._build_absorption_vector(transitions, vector_range, num_bins)
                else:
                    y = None

                if y is not None:
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
                if self.pre_transform:
                    data = self.pre_transform(data)
                data_list.append(data)

            except Exception as e:
                print(f"Skipping row {i}: {e}")
                continue

        if not data_list:
            raise RuntimeError("No valid molecules processed.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

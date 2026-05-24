import os
import math
from pathlib import Path
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
        filter_type,
        block_3_only: bool = False,
        mark_block_3: bool = False,
        y_columns: Optional[Sequence[str]] = None,
        extra_fields: Optional[Sequence[str]] = None,
        transform=None,
        pre_transform=None,
        force_reprocess: bool = False,
        version: str = "v1",
        prediction_type: str = "pairs",
        prediction_params: Optional[dict] = None,
        vis_range: Tuple[float, float] = (380.0, 750.0),
        num_states: int = 10,
        filter_f_value: float = 0.001,
        min_f_value: float = 0.001,
        lorenzian: bool = False,
        sort_by_max_f: bool = True,
        standarize_lambda: bool = False,
        standarize_f: bool = False,
        lambda_bucket_size: int = 0,
        load_representations: str = '',
        lambda_outlier_threshold : str = None,
        f_outlier_threshold : float = None,
        outlier_strategy: float = None,
        convert_to_ev: bool = False


    ):  
        self.filter_type = filter_type
        self.y_columns = list(y_columns) if y_columns else []
        self.extra_fields = list(extra_fields) if extra_fields else []
        self.block_3_only = block_3_only
        self.version = str(version)

        self.prediction_type = prediction_type
        self.prediction_params = prediction_params or {}
        self.min_lambda, self.max_lambda = vis_range
        self.num_states = num_states
        self.min_f_value = min_f_value
        self.filter_f_value = filter_f_value
        self.lorenzian = lorenzian
        self.sort_by_max_f = sort_by_max_f
        self.standarize_lambda = standarize_lambda
        self.standarize_f = standarize_f
        self.lambda_bucket_size = lambda_bucket_size
        self.load_representations = load_representations
        self.mark_block_3 = mark_block_3
        self.lambda_outlier_threshold = lambda_outlier_threshold
        self.f_outlier_threshold = f_outlier_threshold
        self.outlier_strategy = outlier_strategy
        self.convert_to_ev = convert_to_ev

        if self.prediction_type not in {"pairs", "vector", 'only_lambdas', 'binary_classification',
                                        'binary_vector_multiclass', 'binary_vector_multilabel',
                                        'lambda_regressor', 'f_regressor'}:
            raise ValueError(f"Invalid prediction_type: {self.prediction_type}")

        
        if self.load_representations:
            from gjepa.utils.precomputed_embeddings import PrecomputedEmbeddings
            self.precomputed_embedings = PrecomputedEmbeddings(Path(self.load_representations))
            print(self.precomputed_embedings)
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

    @staticmethod
    def serialize_params_for_filename(params: dict) -> str:
        """Convert a dict of parameters into a short, filename-safe string."""
        if not params:
            return "none"
        parts = []
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple)):
                v = "-".join(map(str, v))
            elif isinstance(v, dict):
                v = "_".join(f"{subk}{subv}" for subk, subv in sorted(v.items()))
            parts.append(f"{k}{v}")
        s = "_".join(parts)
        return s.replace(" ", "_").replace("/", "-").replace(":", "_")

    @property
    def processed_file_names(self) -> list[str]:
        pred_type = self.prediction_type.replace(" ", "_")
        params_str = self.serialize_params_for_filename(self.prediction_params)

        vis_range_str = f"{self.min_lambda}-{self.max_lambda}"
        y_str = f"y{len(self.y_columns)}"
        pre_transform = (
            repr(self.pre_transform.__class__.__name__)
            if self.pre_transform
            else "none"
        )
        load_reprs_str =  os.path.basename(self.load_representations) if self.load_representations else 'False'

        filename = (
            f"tmqmg_block3-{self.block_3_only}_"
            f"{pred_type}_{params_str}_"
            f"num_states-{self.num_states}_"
            f"filter_type-{self.filter_type}_min_f_val{self.min_f_value}_"
            f"filter_f_value-{self.filter_f_value}_"
            f"lanbda_bucket_size-{self.lambda_bucket_size}_"
            f"mark-block3-{self.mark_block_3}_"
            f"outlier_str-{self.outlier_strategy}_"
            f"lambda_outlier_thr-{self.lambda_outlier_threshold}_"
            f"f_outlier_thr-{self.f_outlier_threshold}_"
            f"con_ev-{self.convert_to_ev}_"
            f"pre{pre_transform}.pt"
        )

        filename = filename.replace("__", "_").replace("..", ".")
        return [filename]

    @staticmethod
    def _get_csv_filename(block_3_only) -> str:
        return "raw/uvvis_final_40k.csv" if block_3_only else "raw/tmqm_all.csv"

    def _csv_filename(self) -> str:
        return self._get_csv_filename(self.block_3_only)

    def download(self):
        pass



    def _select_all_transitions(self, row: pd.Series) -> List[Tuple[float, float]]:

        candidates: List[Tuple[float, float, int]] = []  # (lambda, f, i)

        for i in range(1,31):
            lam_col = f"lambda_{i}_gasphase"
            f_col = f"f_{i}_gasphase"

            lam = row[lam_col]
            f = row[f_col]
            lam = float(lam)
            f = float(f)

            if f <= self.filter_f_value:
                continue

            candidates.append((lam, f, i))

        if not candidates or len(candidates) < self.num_states:
            return []

        if self.sort_by_max_f:
            candidates_sorted = sorted(candidates, key=lambda x: (-x[1], x[2]))
            chosen = candidates_sorted
        else:
            candidates_sorted = sorted(candidates, key=lambda x: x[2])
            chosen = candidates_sorted

        result = [(lam, f) for (lam, f, _) in chosen]
        return result
    

    def _filter_at_least_one_visible_transition(self, row: pd.Series) -> List[Tuple[float, float]]:

        candidates: List[Tuple[float, float, int]] = []
        has_visible = False
        visible_lambda = None
        visible_f = None
        for i in range(1, 31):
            lam_col = f"lambda_{i}_gasphase"
            f_col = f"f_{i}_gasphase"

            lam = row[lam_col]
            f = row[f_col]
            lam = float(lam)
            f = float(f)

            if f > self.filter_f_value:
                candidates.append((lam, f, i))

                if self.min_lambda <= lam <= self.max_lambda and f > self.min_f_value :
                    has_visible = True
                    visible_lambda = lam
                    visible_f = f

        if not has_visible:
            return []

        if not candidates or len(candidates) < self.num_states:
            return []

        if self.sort_by_max_f:
            candidates_sorted = sorted(candidates, key=lambda x: (-x[1], x[2]))
            chosen = candidates_sorted
        else:
            candidates_sorted = sorted(candidates, key=lambda x: x[2])
            chosen = candidates_sorted

        result = [(lam, f) for (lam, f, _) in chosen if lam != visible_lambda]
        result.insert(0, (visible_lambda, visible_f))
        return result


    
    def filter_data_with_criterion(self, row: pd.Series, filter_type: str):
        if filter_type == 'one_visible_lambda':
            return self._filter_at_least_one_visible_transition(row)
        elif filter_type == 'all_samples':
            return self._select_all_transitions(row)
        else: 
            raise ValueError()

    def _build_top_pairs(self, transitions: List[Tuple[float, float]], num_pairs: int = 10) -> torch.Tensor:

        vec = []
        for lam, f in transitions:
            if len(vec) < num_pairs * 2:
                vec.extend([lam, f])
        return torch.tensor([vec], dtype=torch.float32)

    def _build_absorption_vector(
        self,
        transitions: List[Tuple[float, float]],
        wavelength_range: Tuple[float, float],
        lorenzian: bool = False,
    ) -> torch.Tensor:
        """Build discrete absorption spectrum as histogram or Lorentzian-broadened vector."""
        start, end = wavelength_range
        num_bins = int(end - start)
        lam_grid = np.linspace(start, end, num_bins)
        hist = np.zeros(num_bins)

        if not lorenzian:
            for lam, f in transitions:
                if start <= lam <= end:
                    idx = int((lam - start) / (end - start) * num_bins)
                    idx = min(max(idx, 0), num_bins - 1)
                    hist[idx] += f
        else:
            gamma = 2.0 
            for lam0, f in transitions:
                hist += f * (1/np.pi) * (gamma / ((lam_grid - lam0)**2 + gamma**2))

        return torch.tensor(hist, dtype=torch.float32).unsqueeze(0)


    def _build_only_lambdas(self, transitions: List[Tuple[float, float]], num_pairs: int = 10) -> torch.Tensor:
        vec = []
        for lam, f in transitions:
            if len(vec) < num_pairs:
                vec.extend([lam])

        return torch.tensor([vec], dtype=torch.float32)

    def _build_lambda_regressor(self, transitions: List[Tuple[float, float]], num_pairs: int = 10) -> torch.Tensor:
        vec = []
        lam, f = transitions[num_pairs]
        vec.append(lam)
        return torch.tensor([vec], dtype=torch.float32)

    def _build_f_regressor(self, transitions: List[Tuple[float, float]], num_pairs: int = 10) -> torch.Tensor:
        vec = []
        lam, f = transitions[num_pairs]
        vec.append(f)
        return torch.tensor([vec], dtype=torch.float32)

    def _build_binary(self, transitions: List[Tuple[float, float]], min_f_value: float = 0) -> torch.Tensor:
        pos = 0
        for lam, f in transitions:
            if self.min_lambda <= lam <= self.max_lambda and f > min_f_value:
                pos = 1
                break
        
        if pos not in [0,1]:
            raise ValueError()
        return torch.tensor([[pos]], dtype=torch.float32)

    def _build_binary_vector_multiclass(self, transitions: List[Tuple[float, float]]) -> torch.Tensor:
        start, end = self.min_lambda, self.max_lambda
        
        if end <= start:
            raise ValueError(f"max_lambda ({end}) musi być > min_lambda ({start}).")
        
        total_range = end - start
        num_bins = math.ceil(total_range / self.lambda_bucket_size)
        
        hist = torch.zeros(num_bins, dtype=torch.int64)
        
        for lam, f in transitions:
            if (start <= lam <= end) and (f > self.min_f_value):
                idx = int((lam - start) / self.lambda_bucket_size)
                
                idx = max(0, min(idx, num_bins - 1))
                hist[idx] = 1
                break

        return torch.tensor([idx], dtype=torch.long)


    def _build_binary_vector_multilabel(self, transitions: List[Tuple[float, float]]) -> torch.Tensor:
        start, end = self.min_lambda, self.max_lambda
        
        if end <= start:
            raise ValueError(f"max_lambda ({end}) musi być > min_lambda ({start}).")
        
        total_range = end - start
        num_bins = math.ceil(total_range / self.lambda_bucket_size)
        
        hist = torch.zeros(num_bins, dtype=torch.int64)
        
        for lam, f in transitions:
            if (start <= lam <= end) and (f > self.min_f_value):
                idx = int((lam - start) / self.lambda_bucket_size)
                idx = max(0, min(idx, num_bins - 1))
                hist[idx] = 1
        return hist.unsqueeze(0)
    
    def _prepare_the_output_format(self, transitions):
        if not self.prediction_type in {"pairs", "vector", "only_lambdas", "binary_classification",
                                        'binary_vector_multiclass', 'binary_vector_multilabel',
                                        'lambda_regressor', 'f_regressor'}:
            raise ValueError('prediction type did not mach: ', " pairs ", " vector ",
                             "only_lambdas", " binary_classification",
                             'lambda_regressor', 'f_regressor')
        if self.prediction_type == "pairs":
            num_pairs = self.num_states
            min_f_value = self.min_f_value
            return self._build_top_pairs(transitions, num_pairs=num_pairs)

        elif self.prediction_type == 'vector':
            vector_range = self.prediction_params["range"]
            return self._build_absorption_vector(transitions, vector_range, self.lorenzian)
        elif self.prediction_type == 'only_lambdas':
            return self._build_only_lambdas(transitions, num_pairs=self.num_states)
        elif self.prediction_type == 'binary_classification':
            y = self._build_binary(transitions, min_f_value=self.min_f_value)
            return y
        elif self.prediction_type == 'binary_vector_multiclass':
            return self._build_binary_vector_multiclass(transitions)
        elif self.prediction_type == 'binary_vector_multilabel':
            return self._build_binary_vector_multilabel(transitions)
        elif self.prediction_type == 'lambda_regressor':
            return self._build_lambda_regressor(transitions, num_pairs=self.num_states)
        elif self.prediction_type == 'f_regressor':
            return self._build_f_regressor(transitions, num_pairs=self.num_states)

    def remove_outliers(self, transitions):
        if not transitions:
            return []
        lambda_outlier_threshold = self.lambda_outlier_threshold
        f_outlier_threshold = self.f_outlier_threshold

        if self.outlier_strategy == "whole-compound-outlier-removal":
            # if any lambda is above the lambda_outlier_threshold, remove whole compound
            if lambda_outlier_threshold is not None and any(lam > lambda_outlier_threshold for lam, f in transitions):
                return []
            

        elif self.outlier_strategy == "remove_outlying_transitions":
            return_transitions = transitions
            if lambda_outlier_threshold is not None:
                return_transitions = [(lam, f) for lam, f in return_transitions if lam < lambda_outlier_threshold]
            if f_outlier_threshold is not None:
                return_transitions = [(lam, min(f, f_outlier_threshold)) for lam, f in return_transitions]
            return return_transitions
        return transitions
        
    def convert_lambdas_to_ev(self, transitions):
        if not transitions:
            return []
        if self.convert_to_ev and self.prediction_type in {"pairs", 'only_lambdas',
                                                           'lambda_regressor'}:
            ev_trainsitions = [(1239.84 / lam, f) for lam, f in transitions]
            return ev_trainsitions
        return transitions

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

        if self.mark_block_3:
            if not self.block_3_only:
                block_3_csv_path = os.path.join(self.root, self._get_csv_filename(block_3_only=True))
                if not os.path.exists(block_3_csv_path):
                    raise FileNotFoundError(f"Block 3 CSV not found for marking: {block_3_csv_path}")

                df_block_3 = pd.read_csv(block_3_csv_path, usecols=["CSD_code"])
                block_3_codes = set(df_block_3["CSD_code"])
                df["is_from_block_3"] = df["CSD_code"].isin(block_3_codes)
            else:
                df["is_from_block_3"] = True

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
        pos_classes_counter = 0
        neg_classes_counter = 0

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            num_atoms = int(row["num_atoms"]) if "num_atoms" in row and not pd.isna(row["num_atoms"]) else None
            pos = _parse_coords(row["atom_coords"], expected_n=num_atoms)
            z = _parse_atom_types(row["atom_types"])
            if pos.size(0) != z.numel():
                raise ValueError(f"Atom count mismatch: pos={pos.size(0)}, z={z.numel()}")
            smiles = "" if pd.isna(row["SMILES"]) else str(row["SMILES"])
            origin_id = None if pd.isna(row["origin_ID"]) else str(row["origin_ID"])
            csd_code = None if pd.isna(row["CSD_code"]) else str(row["CSD_code"])
            kwargs = dict(pos=pos, z=z, smiles=smiles, origin_id=origin_id, CSD_code=csd_code)

            if self.mark_block_3:
                kwargs["is_from_block_3"] = row["is_from_block_3"]

            if self.load_representations:
                emb = self.precomputed_embedings.get_embedding(csd_code)
                kwargs['representation'] = emb

            transitions = self.filter_data_with_criterion(row, self.filter_type)
            transitions = self.remove_outliers(transitions)
            if not transitions:
                continue
            transitions = self.convert_lambdas_to_ev(transitions)
            if not transitions:
                continue
            y = self._prepare_the_output_format(transitions)
            if y is not None and self.prediction_type == 'binary_classification':
                if y.item() == 1:
                    pos_classes_counter += 1
                elif y.item() == 0:
                    neg_classes_counter += 1
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

        print(f"Positive classes: {pos_classes_counter}, Negative classes: {neg_classes_counter}")
        if not data_list:
            raise RuntimeError("No valid molecules processed.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

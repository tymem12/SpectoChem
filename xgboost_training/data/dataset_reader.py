import os
import numpy as np
import pandas as pd
from typing import Optional, List, Set, Dict, Any, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch_geometric.data import Data

from data.value_capper import ValueCapper
from data.target_generator import TargetGenerator
from data.multilabel_generator import MultilabelGenerator
from data.binary_pairs_generator import BinaryPairsGenerator


class DatasetReaderFromGraphModule:
    """
    DatasetReader that extracts molecular data directly from GraphLevelDataModule.
    No CSV matching needed - uses the exact same molecules as the graph module.
    """

    def __init__(self, config):
        """
        Args:
            config: Config object with hydra_config_dir and hydra_overrides
        """
        self.csv_path = config.data['csv_path']
        self.value_capper = ValueCapper(config)
        self.target_generator = TargetGenerator(config)

        self.hydra_config_dir = config.hydra_config_dir
        self.hydra_overrides = config.hydra_overrides

        print(f"  DatasetReader initialized with:")
        print(f"    hydra_config_dir: {self.hydra_config_dir}")
        print(f"    hydra_overrides: {self.hydra_overrides}")

        multilabel_config = config._config.get('multilabel')
        self.random_state = getattr(config, 'random_state', 42)

        self.multilabel_generator = MultilabelGenerator(config) if multilabel_config else None

        binary_pairs_config = config._config.get('binary_pairs')
        self.binary_pairs_generator = BinaryPairsGenerator(config) if binary_pairs_config else None

        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.target_pairs = None
        self.binary_target_name = None
        self.bucket_columns = None
        self.binary_f_columns = None

    def _has_dataset_override(self, overrides):
        """Check if user provided any dataset-related override."""
        return any(o.startswith('dataset=') or o.startswith('dataset.') for o in overrides)

    def _has_model_override(self, overrides):
        """Check if user provided any model-related override."""
        return any(o.startswith('model=') or o.startswith('model.') for o in overrides)

    def _has_backbone_override(self, overrides):
        """Check if user provided any backbone-related override."""
        return any('backbone' in o for o in overrides)

    def _load_graph_level_datamodule(self) -> Any:
        """Load GraphLevelDataModule using Hydra."""
        config_dir = Path(self.hydra_config_dir).resolve()

        print(f"\nLoading GraphLevelDataModule via Hydra...")
        print(f"  Config directory: {config_dir}")

        if not config_dir.exists():
            raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")

        config_yaml = config_dir / "config.yaml"
        if not config_yaml.exists():
            raise FileNotFoundError(f"config.yaml not found in {config_dir}")

        overrides = self.hydra_overrides.copy() if self.hydra_overrides else []
        print(f"  Overrides from config: {overrides}")

        if not self._has_model_override(overrides):
            overrides.append('model=supervised_graph_level')
            print(f"  [ADDED] model=supervised_graph_level")
        else:
            print(f"  [SKIPPED] model= (user provided: {[o for o in overrides if 'model' in o][0]})")

        if not self._has_backbone_override(overrides):
            overrides.append('backbone@model.backbone=schnet')
            print(f"  [ADDED] backbone@model.backbone=schnet")
        else:
            print(f"  [SKIPPED] backbone (user provided: {[o for o in overrides if 'backbone' in o][0]})")

        if not self._has_dataset_override(overrides):
            task = getattr(self.target_generator, 'task', 'binary')
            if 'pairs' in task:
                overrides.append('dataset=TMQM_SPECTO_PAIRS')
                print(f"  [ADDED] dataset=TMQM_SPECTO_PAIRS")
            elif 'multilabel' in task:
                overrides.append('dataset=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL')
                print(f"  [ADDED] dataset=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL")
            else:
                overrides.append('dataset=TMQM_SPECTO_BINARY')
                print(f"  [ADDED] dataset=TMQM_SPECTO_BINARY")
        else:
            dataset_override = [o for o in overrides if 'dataset' in o][0]
            print(f"  [SKIPPED] dataset= (user provided: {dataset_override})")

        if not any(o.startswith('training.random_seed') for o in overrides):
            overrides.append(f"training.random_seed={self.random_state}")
            print(f"  [ADDED] training.random_seed={self.random_state}")

        print(f"  Final overrides: {overrides}")

        with hydra.initialize_config_dir(
            config_dir=str(config_dir),
            version_base="1.3"
        ):
            cfg = hydra.compose(config_name="config", overrides=overrides)

            print(f"  Loaded Hydra config with dataset: {cfg.dataset.name}")

            if hasattr(cfg.dataset, 'block_3_split_mode'):
                print(f"  Dataset block_3_split_mode: {cfg.dataset.block_3_split_mode}")
            if hasattr(cfg, 'training') and hasattr(cfg.training, 'random_seed'):
                print(f"  Training random_seed: {cfg.training.random_seed}")

            from gjepa.datasets.graph_level import GraphLevelDataModule

            class DatasetConfigWrapper:
                def __init__(self, dict_config):
                    self._config = dict_config
                    self._converted_params = None

                def __getattr__(self, name):
                    if name == 'additional_loading_params':
                        if self._converted_params is None and hasattr(self._config, 'additional_loading_params'):
                            params = self._config.additional_loading_params
                            if params is not None:
                                from omegaconf import OmegaConf
                                self._converted_params = OmegaConf.to_container(params, resolve=True)
                            else:
                                self._converted_params = None
                        return self._converted_params
                    return getattr(self._config, name)

                def __getitem__(self, key):
                    return self._config[key]

                def __contains__(self, key):
                    return key in self._config

                def get(self, key, default=None):
                    return self._config.get(key, default)

            if isinstance(cfg.dataset, DictConfig):
                dataset_config = DatasetConfigWrapper(cfg.dataset)
            else:
                dataset_config = cfg.dataset

            datamodule = GraphLevelDataModule(
                dataset_config=dataset_config,
                batch_size=cfg.training.batch_size if hasattr(cfg, 'training') else 32,
                pos_enc_path=cfg.pos_encoding.file if hasattr(cfg, 'pos_encoding') and cfg.pos_encoding else None,
            )

            datamodule.setup(stage='fit')

            return datamodule

    def _extract_data_from_subset(self, subset) -> List[Dict]:
        """Extract molecular data from a Subset dataset."""
        molecules = []

        if hasattr(subset, 'indices'):
            indices = subset.indices
            base_dataset = subset.dataset
        else:
            indices = range(len(subset))
            base_dataset = subset

        print(f"    Subset has {len(indices)} indices")

        for idx in indices:
            data = base_dataset[idx]

            mol_id = self._extract_molecule_id(data)
            if not mol_id:
                continue

            atom_types = self._extract_atom_types(data)
            atom_coords = self._extract_atom_coords(data)

            if not atom_types or not atom_coords:
                continue

            y = self._extract_targets(data)

            molecules.append({
                'CSD_code': mol_id,
                'atom_types': atom_types,
                'atom_coords': atom_coords,
                'y': y,
            })

        return molecules

    def _extract_molecule_id(self, data: Data) -> Optional[str]:
        """Extract molecule identifier from PyG Data object."""
        for attr in ['name', 'CSD_code', 'id', 'subset', 'mol_id']:
            if hasattr(data, attr):
                val = getattr(data, attr)
                if val is not None:
                    if isinstance(val, str):
                        return val.strip().upper()
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        return str(val[0]).strip().upper()
                    elif hasattr(val, 'item'):
                        return str(val.item()).strip().upper()

        if hasattr(data, 'to_dict'):
            d = data.to_dict()
            for key in ['name', 'CSD_code', 'id']:
                if key in d:
                    return str(d[key]).strip().upper()

        return None

    def _extract_atom_types(self, data: Data) -> List[str]:
        """Extract atom types from PyG Data object."""
        atom_types = []

        if hasattr(data, 'z') and data.z is not None:
            atomic_nums = data.z.cpu().numpy() if hasattr(data.z, 'cpu') else data.z
            num_to_symbol = {
                1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S',
                17: 'Cl', 35: 'Br', 53: 'I', 5: 'B', 14: 'Si', 34: 'Se', 33: 'As'
            }
            for num in atomic_nums:
                symbol = num_to_symbol.get(int(num), 'X')
                atom_types.append(symbol)

        elif hasattr(data, 'x') and data.x is not None:
            x = data.x.cpu().numpy() if hasattr(data.x, 'cpu') else data.x
            element_list = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'As']
            for row in x:
                idx = np.argmax(row)
                if idx < len(element_list):
                    atom_types.append(element_list[idx])
                else:
                    atom_types.append('X')

        return atom_types

    def _extract_atom_coords(self, data: Data) -> List[List[float]]:
        """Extract atom coordinates from PyG Data object."""
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos.cpu().numpy() if hasattr(data.pos, 'cpu') else data.pos
            return pos.tolist()
        return []

    def _extract_targets(self, data: Data) -> Any:
        """Extract target values from PyG Data object."""
        if hasattr(data, 'y') and data.y is not None:
            y = data.y.cpu().numpy() if hasattr(data.y, 'cpu') else data.y
            if hasattr(y, 'flatten'):
                y = y.flatten()
            return y
        return None

    def _create_dataframe_from_molecules(self, molecules: List[Dict], prediction_type: str = 'binary_classification') -> pd.DataFrame:
        """Create DataFrame from extracted molecule data."""
        df = pd.DataFrame(molecules)

        df['atom_types'] = df['atom_types'].apply(lambda x: str(x))
        df['atom_coords'] = df['atom_coords'].apply(lambda x: str(x))

        if 'y' in df.columns and len(df) > 0:
            first_y = df['y'].iloc[0]
            if first_y is not None:
                y_len = len(first_y) if hasattr(first_y, '__len__') else 1

                if prediction_type == 'pairs':
                    for i in range(0, y_len, 2):
                        pair_idx = i // 2 + 1
                        df[f'lambda_{pair_idx}_gasphase'] = df['y'].apply(lambda y: y[i] if y is not None and len(y) > i else 0)
                        df[f'f_{pair_idx}_gasphase'] = df['y'].apply(lambda y: y[i+1] if y is not None and len(y) > i+1 else 0)

                elif prediction_type == 'vector':
                    for i in range(y_len):
                        df[f'f_{i+1}_gasphase'] = df['y'].apply(lambda y: y[i] if y is not None and len(y) > i else 0)

                elif prediction_type == 'only_lambdas':
                    for i in range(y_len):
                        df[f'lambda_{i+1}_gasphase'] = df['y'].apply(lambda y: y[i] if y is not None and len(y) > i else 0)

                elif prediction_type in ['binary_vector_multiclass', 'binary_vector_multilabel']:
                    for i in range(y_len):
                        df[f'abs_bucket_{i}'] = df['y'].apply(lambda y: int(y[i]) if y is not None and len(y) > i else 0)

                elif prediction_type == 'binary_classification':
                    df['has_uvvis_peak'] = df['y'].apply(lambda y: int(y[0]) if y is not None and len(y) > 0 else 0)

        return df

    def load_and_prepare(self):
        """Load data directly from GraphLevelDataModule."""
        print(f"\n{'='*70}")
        print(f"LOADING DATA FROM GRAPHLEVELDATAMODULE (NO CSV MATCHING)")
        print(f"{'='*70}")

        datamodule = self._load_graph_level_datamodule()

        prediction_type = 'binary_classification'
        if hasattr(datamodule, 'config') and hasattr(datamodule.config, 'additional_loading_params'):
            params = datamodule.config.additional_loading_params
            if isinstance(params, dict):
                prediction_type = params.get('prediction_type', 'binary_classification')
            else:
                prediction_type = getattr(params, 'prediction_type', 'binary_classification')

        print(f"  Detected prediction type: {prediction_type}")

        print(f"\nExtracting molecules from GraphLevelDataModule...")

        train_molecules = []
        val_molecules = []
        test_molecules = []

        if hasattr(datamodule, 'train_ds') and datamodule.train_ds is not None:
            print(f"  Processing train split...")
            train_molecules = self._extract_data_from_subset(datamodule.train_ds)
            print(f"    Extracted {len(train_molecules)} molecules")

        if hasattr(datamodule, 'val_ds') and datamodule.val_ds is not None:
            print(f"  Processing val split...")
            val_molecules = self._extract_data_from_subset(datamodule.val_ds)
            print(f"    Extracted {len(val_molecules)} molecules")

        if hasattr(datamodule, 'test_ds') and datamodule.test_ds is not None:
            print(f"  Processing test split...")
            test_molecules = self._extract_data_from_subset(datamodule.test_ds)
            print(f"    Extracted {len(test_molecules)} molecules")

        self.train_df = self._create_dataframe_from_molecules(train_molecules, prediction_type)
        self.val_df = self._create_dataframe_from_molecules(val_molecules, prediction_type)
        self.test_df = self._create_dataframe_from_molecules(test_molecules, prediction_type)

        print(f"\n{'='*70}")
        print(f"EXTRACTED DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Train: {len(self.train_df)} molecules")
        print(f"Val: {len(self.val_df)} molecules")
        print(f"Test: {len(self.test_df)} molecules")
        print(f"{'='*70}\n")

        self._generate_targets_from_csv()

        return self._get_data_dict()

    def _generate_targets_from_csv(self):
        """Generate targets by loading CSV and matching to extracted molecules."""
        print(f"Loading target data from {self.csv_path}...")
        df_base = pd.read_csv(self.csv_path, index_col=0)
        df_star = self._load_star_data()
        df_full = self._merge_dataframes(df_base, df_star)

        train_ids = set(self.train_df['CSD_code'])
        val_ids = set(self.val_df['CSD_code'])
        test_ids = set(self.test_df['CSD_code'])
        all_graph_ids = train_ids | val_ids | test_ids

        df_full['CSD_code_norm'] = df_full['CSD_code'].astype(str).str.strip().str.upper()
        df_filtered = df_full[df_full['CSD_code_norm'].isin(all_graph_ids)].copy()

        print(f"Matched {len(df_filtered)}/{len(all_graph_ids)} molecules from graph to CSV")

        self.target_pairs = self.target_generator.get_limited_pairs(df_filtered)

        if self.target_generator.should_run_pairs():
            print(f"\nPairs task enabled - using {len(self.target_pairs)} pairs")

        if self.target_generator.should_run_binary():
            df_filtered, self.binary_target_name = self.target_generator.create_binary_target(df_filtered)

        if self.target_generator.should_run_binary_pairs():
            if self.binary_pairs_generator is None:
                raise ValueError("Binary-pairs task requested but 'binary_pairs' section missing")
            df_filtered, self.binary_f_columns = self.binary_pairs_generator.create_binary_f_targets(
                df_filtered, self.target_pairs
            )

        if self.target_generator.should_run_multilabel():
            if self.multilabel_generator is None:
                raise ValueError("Multilabel task requested but 'multilabel' section missing")
            df_filtered, self.bucket_columns = self.multilabel_generator.create_bucket_targets(df_filtered)

        for split_name, split_df in [('train', self.train_df), ('val', self.val_df), ('test', self.test_df)]:
            split_df['CSD_code_norm'] = split_df['CSD_code'].astype(str).str.strip().str.upper()

            merged = split_df.merge(
                df_filtered,
                left_on='CSD_code_norm',
                right_on='CSD_code_norm',
                how='left',
                suffixes=('', '_csv')
            )

            for col in df_filtered.columns:
                if col not in ['CSD_code', 'CSD_code_norm', 'atom_types', 'atom_coords']:
                    if col in merged.columns:
                        split_df[col] = merged[col].values

            split_df.drop(columns=['CSD_code_norm'], inplace=True, errors='ignore')

        print(f"\nTarget generation complete")

    def _load_star_data(self):
        base_dir = os.path.dirname(self.csv_path)
        star_path = os.path.join(base_dir, "tmqmg_star.csv")

        if not os.path.exists(star_path):
            raise FileNotFoundError(f"tmqmg_star.csv not found at {star_path}")

        print(f"Loading TMQMG star data from {star_path}...")
        return pd.read_csv(star_path)

    def _merge_dataframes(self, df_base, df_star):
        if "CSD_code" not in df_base.columns:
            raise KeyError("Missing 'CSD_code' column in base CSV.")
        if "id" not in df_star.columns:
            raise KeyError("Missing 'id' column in tmqmg_star.csv.")

        overlap = set(df_base.columns) & set(df_star.columns) - {"CSD_code", "id"}
        if overlap:
            raise ValueError(f"Overlapping columns: {sorted(overlap)}")

        df = pd.merge(df_base, df_star, how="inner", left_on="CSD_code", right_on="id")
        print(f"Merged dataset: {len(df)} rows")
        return df

    def get_target_pairs(self):
        return self.target_pairs

    def get_binary_target_name(self):
        return self.binary_target_name

    def get_bucket_columns(self):
        return self.bucket_columns

    def get_binary_f_columns(self):
        return self.binary_f_columns

    def _get_data_dict(self):
        return {
            'train_df': self.train_df,
            'test_df': self.test_df,
            'val_df': self.val_df
        }

    def get_combined_atom_data(self):
        all_atom_types = np.concatenate([
            self.train_df['atom_types'].values,
            self.test_df['atom_types'].values,
            self.val_df['atom_types'].values
        ])
        all_atom_coords = np.concatenate([
            self.train_df['atom_coords'].values,
            self.test_df['atom_coords'].values,
            self.val_df['atom_coords'].values
        ])
        return all_atom_types, all_atom_coords

    def get_feature_splits(self, descriptors):
        n_train = len(self.train_df)
        n_test = len(self.test_df)

        X_train = descriptors[:n_train]
        X_test = descriptors[n_train:n_train + n_test]
        X_val = descriptors[n_train + n_test:]

        return X_train, X_test, X_val
#!/usr/bin/env python3
"""
Complete Benchmark Script for Molecular Property Prediction

Features:
- Single --config argument
- Hydra-style overrides for GraphLevelDataModule
- Experiments defined in config.yaml
- Runs binary, pairs, and multilabel tasks
- Generates descriptors once, reuses for all tasks
- Proper result saving for each task type
- Per-experiment random_state support

Usage:
    PYTHONPATH=. python xgboost_training/benchmark.py \\
        --config xgboost_training/config.yaml \\
        dataset.block_3_split_mode=train \\
        training.random_seed=42
"""

import argparse
import warnings
import logging
import os
import sys
import json
import yaml
import traceback
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config
from data.dataset_reader import DatasetReaderFromGraphModule as DatasetReader
from descriptors.factory import DescriptorFactory
from models.trainer import XGBoostTrainer
from models.classifier import XGBoostClassifier
from models.multilabel_trainer import MultilabelTrainer
from models.binary_pairs_trainer import BinaryPairsTrainer
from utils.metrics import ResultsSaver


def setup_logging():
    """Setup logging to suppress warnings."""
    warnings.filterwarnings('ignore')
    logging.getLogger('rdkit').setLevel(logging.CRITICAL)
    RDLogger.DisableLog('rdApp.*')


class BenchmarkRunner:
    """Runner for comprehensive benchmark experiments."""

    def __init__(self, base_config_path: str, hydra_overrides: List[str]):
        """
        Initialize benchmark runner.

        Args:
            base_config_path: Path to base configuration file
            hydra_overrides: List of Hydra override strings
        """
        self.base_config_path = base_config_path
        self.hydra_overrides = hydra_overrides

        with open(base_config_path, 'r') as f:
            self.base_config_dict = yaml.safe_load(f)

        self.experiments_config = self.base_config_dict.get('experiments', [])
        if not self.experiments_config:
            raise ValueError("No experiments defined in config. Add 'experiments' list to config.yaml")

        self.base_output_dir = self.base_config_dict.get('output', {}).get(
            'results_dir', './xgboost_results'
        ).replace('results', 'benchmark_results')

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.base_output_dir, f'benchmark_{self.timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        base_config_backup_path = os.path.join(self.run_dir, 'base_config_backup.yaml')
        with open(base_config_backup_path, 'w') as f:
            yaml.dump(self.base_config_dict, f, default_flow_style=False)

        self.summary = {
            'completed': [],
            'failed': [],
            'skipped': [],
            'base_config_path': base_config_path,
            'base_config_backup': base_config_backup_path,
            'hydra_overrides': hydra_overrides,
            'experiments_count': len(self.experiments_config),
            'timestamp': self.timestamp
        }

        print(f"Benchmark run directory: {self.run_dir}")
        print(f"Number of experiments: {len(self.experiments_config)}")
        print(f"User Hydra overrides: {hydra_overrides if hydra_overrides else 'None'}")

    def _has_dataset_override(self, overrides: List[str]) -> bool:
        """Check if user provided any dataset-related override."""
        return any(o.startswith('dataset=') or o.startswith('dataset.') for o in overrides)

    def _has_model_override(self, overrides: List[str]) -> bool:
        """Check if user provided any model-related override."""
        return any(o.startswith('model=') or o.startswith('model.') for o in overrides)

    def _has_backbone_override(self, overrides: List[str]) -> bool:
        """Check if user provided any backbone-related override."""
        return any('backbone' in o for o in overrides)

    def _get_hydra_overrides_for_task(self, task: str) -> List[str]:
        """
        Build complete Hydra overrides for a specific task.

        Preserves user overrides and only adds defaults if needed.
        """
        overrides = self.hydra_overrides.copy()

        print(f"\\n  Building overrides for task '{task}':")
        print(f"    Starting with user overrides: {overrides}")

        task_to_dataset_name = {
            'binary': 'TMQM_SPECTO_BINARY',
            'pairs': 'TMQM_SPECTO_PAIRS',
            'binary-pairs': 'TMQM_SPECTO_PAIRS',
            'multilabel': 'TMQM_SPECTO_BINARY_VECTOR_MULTILABEL',
        }

        if not self._has_dataset_override(overrides):
            dataset_name = task_to_dataset_name.get(task, 'TMQM_SPECTO_BINARY')
            overrides.append(f'dataset={dataset_name}')
            print(f"    [ADDED] dataset={dataset_name}")
        else:
            dataset_overrides = [o for o in overrides if 'dataset' in o]
            print(f"    [SKIPPED] dataset= (user provided: {dataset_overrides})")

        if not self._has_model_override(overrides):
            overrides.append('model=supervised_graph_level')
            print(f"    [ADDED] model=supervised_graph_level")
        else:
            print(f"    [SKIPPED] model= (user provided)")

        if not self._has_backbone_override(overrides):
            overrides.append('backbone@model.backbone=schnet')
            print(f"    [ADDED] backbone@model.backbone=schnet")
        else:
            print(f"    [SKIPPED] backbone (user provided)")

        task_to_hydra_exp = self.base_config_dict.get('task_to_hydra_exp', {})
        if task in task_to_hydra_exp:
            exp_override = task_to_hydra_exp[task]
            if exp_override and exp_override not in overrides:
                overrides.append(exp_override)
                print(f"    [ADDED] {exp_override}")

        print(f"    Final overrides: {overrides}")
        return overrides

    def generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from config."""
        experiments = []

        for idx, exp_config in enumerate(self.experiments_config):
            task = exp_config.get('task', 'binary')
            model_type = exp_config.get('model_type', 'xgboost')
            descriptor_type = exp_config.get('descriptor_type', 'soap')
            remove_outlier = exp_config.get('remove_outlier', False)
            tune = exp_config.get('tune', False)
            random_state = exp_config.get('random_state', self.base_config_dict.get('random_state', 42))

            if task == 'multilabel' and model_type == 'svm':
                print(f"Skipping experiment {idx}: multilabel with SVM not supported")
                self.summary['skipped'].append({
                    'index': idx,
                    'reason': 'multilabel with SVM not supported',
                    'config': exp_config
                })
                continue

            exp_id = f"{task}_{model_type}_{descriptor_type}_outlier{remove_outlier}_tune{tune}_seed{random_state}"
            hydra_overrides = self._get_hydra_overrides_for_task(task)

            experiment = {
                'id': exp_id,
                'task': task,
                'model_type': model_type,
                'descriptor_type': descriptor_type,
                'remove_outlier': remove_outlier,
                'tune': tune,
                'random_state': random_state,
                'hydra_overrides': hydra_overrides,
                'output_dir': os.path.join(self.run_dir, exp_id)
            }
            experiments.append(experiment)

        return experiments

    def create_modified_config(self, exp_config: Dict[str, Any]) -> str:
        """
        Create a modified config file for the experiment.

        Returns path to temporary config file.
        """
        config_dict = copy.deepcopy(self.base_config_dict)

        config_dict['task'] = exp_config['task']
        config_dict['model_type'] = exp_config['model_type']
        config_dict['descriptor']['type'] = exp_config['descriptor_type']
        # Set per-experiment random_state
        config_dict['random_state'] = exp_config['random_state']

        if exp_config['remove_outlier']:
            config_dict['data']['value_caps'] = {'lambda_max': 1500, 'f_max': 0.3}

        if exp_config['tune']:
            model_type = exp_config['model_type']
            if 'tuning' in config_dict and model_type in config_dict['tuning']:
                config_dict['tuning'][model_type]['enabled'] = True

            task = exp_config['task']
            task_tuning_key = task.replace('-', '_')
            if 'tuning' in config_dict and task_tuning_key in config_dict['tuning']:
                config_dict['tuning'][task_tuning_key]['enabled'] = True

        config_dict['output']['models_dir'] = os.path.join(exp_config['output_dir'], 'models')
        config_dict['output']['results_dir'] = os.path.join(exp_config['output_dir'], 'results')

        config_dict['hydra_config_dir'] = './config'
        config_dict['hydra_overrides'] = exp_config['hydra_overrides']

        if 'custom_tasks' in config_dict:
            if exp_config['task'] == 'all':
                for key in config_dict['custom_tasks']:
                    config_dict['custom_tasks'][key] = True
            elif exp_config['task'] == 'custom':
                pass
            elif exp_config['task'] == 'binary-pairs':
                config_dict['custom_tasks']['pairs'] = True
                config_dict['custom_tasks']['binary_pairs'] = True
                config_dict['custom_tasks']['binary'] = False
                config_dict['custom_tasks']['multilabel'] = False
            else:
                for key in config_dict['custom_tasks']:
                    config_dict['custom_tasks'][key] = (key == exp_config['task'])

        os.makedirs(exp_config['output_dir'], exist_ok=True)
        temp_config_path = os.path.join(exp_config['output_dir'], 'config.yaml')

        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        override_summary = {
            'experiment_id': exp_config['id'],
            'overrides': {
                'task': exp_config['task'],
                'model_type': exp_config['model_type'],
                'descriptor_type': exp_config['descriptor_type'],
                'remove_outlier': exp_config['remove_outlier'],
                'tune': exp_config['tune'],
                'random_state': exp_config['random_state'],
                'hydra_overrides': exp_config['hydra_overrides'],
            },
            'preserved_settings': {
                'gpu': config_dict.get('gpu', {}),
                'model_hyperparams': {
                    'xgboost': {k: v for k, v in config_dict.get('xgboost', {}).items() if k != 'device'},
                    'random_forest': config_dict.get('random_forest', {}),
                    'svm': config_dict.get('svm', {})
                },
                'data_paths': {k: v for k, v in config_dict.get('data', {}).items() if k != 'value_caps'},
            }
        }

        summary_path = os.path.join(exp_config['output_dir'], 'config_override_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(override_summary, f, indent=2)

        return temp_config_path

    def run_single_experiment(self, exp_config: Dict[str, Any]) -> bool:
        """
        Run a single experiment.

        Returns True if successful, False otherwise.
        """
        exp_id = exp_config['id']
        output_dir = exp_config['output_dir']

        print(f"\\n{'='*80}")
        print(f"Running Experiment: {exp_id}")
        print(f"{'='*80}")
        print(f"  Task: {exp_config['task']}")
        print(f"  Model: {exp_config['model_type']}")
        print(f"  Descriptor: {exp_config['descriptor_type']}")
        print(f"  Remove Outliers: {exp_config['remove_outlier']}")
        print(f"  Tuning: {exp_config['tune']}")
        print(f"  Random State: {exp_config['random_state']}")
        print(f"  Output: {output_dir}")


        try:
            temp_config_path = self.create_modified_config(exp_config)
            config = load_config(temp_config_path)

            results = self._execute_experiment(config, exp_config)

            metadata = {
                'experiment_id': exp_id,
                'config': exp_config,
                'results_summary': results,
                'completed_at': datetime.now().isoformat(),
                'status': 'completed',
                'base_config_path': self.base_config_path
            }

            metadata_path = os.path.join(output_dir, 'experiment_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\\n  Experiment {exp_id} completed successfully!")
            print(f"  Results saved to: {output_dir}")

            self.summary['completed'].append({
                'id': exp_id,
                'output_dir': output_dir,
                'results': results
            })

            return True

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print(f"\\n  Experiment {exp_id} FAILED!")
            print(f"  Error: {error_msg}")
            if os.getenv('DEBUG'):
                print(f"\\nTraceback:\\n{error_trace}")

            error_info = {
                'experiment_id': exp_id,
                'config': exp_config,
                'error': error_msg,
                'traceback': error_trace,
                'failed_at': datetime.now().isoformat(),
                'status': 'failed',
                'base_config_path': self.base_config_path
            }

            os.makedirs(output_dir, exist_ok=True)
            error_path = os.path.join(output_dir, 'error_info.json')
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)

            self.summary['failed'].append({
                'id': exp_id,
                'error': error_msg,
                'output_dir': output_dir
            })

            return False

    def _execute_experiment(self, config, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual experiment logic.

        This includes:
        1. Loading data from GraphLevelDataModule
        2. Generating descriptors
        3. Running all applicable tasks (binary, pairs, multilabel)
        4. Saving results
        """
        results_summary = {
            'task': config.task,
            'model_type': exp_config['model_type'],
            'descriptor_type': exp_config['descriptor_type'],
            'random_state': exp_config['random_state'],
            'datasets': {},
            'tasks_run': []
        }

        print(f"\\nStep 1: Loading and preparing data...")
        data_reader = DatasetReader(config)
        data_dict = data_reader.load_and_prepare()

        results_summary['datasets'] = {
            'train_size': len(data_reader.train_df),
            'test_size': len(data_reader.test_df),
            'val_size': len(data_reader.val_df)
        }

        print(f"  Dataset sizes:")
        print(f"    Train: {results_summary['datasets']['train_size']}")
        print(f"    Val: {results_summary['datasets']['val_size']}")
        print(f"    Test: {results_summary['datasets']['test_size']}")

        print(f"\\nStep 2: Generating {exp_config['descriptor_type']} descriptors...")
        descriptor_gen = DescriptorFactory.create(config)

        all_atom_types, all_atom_coords = data_reader.get_combined_atom_data()
        descriptor_gen.fit(all_atom_types)
        descriptors = descriptor_gen.transform(all_atom_types, all_atom_coords)
        descriptors = descriptors.astype(np.float32)

        results_summary['descriptor_shape'] = descriptors.shape
        print(f"  Descriptors shape: {descriptors.shape}")

        saver = ResultsSaver(config)

        task = exp_config['task']

        if task in ['pairs', 'binary-pairs', 'all']:
            print(f"\\nStep 3: Training regression models for pairs...")
            target_pairs = data_reader.get_target_pairs()
            if target_pairs:
                trainer = XGBoostTrainer(config)
                pairs_results = trainer.train(data_reader, descriptors, target_pairs)
                saver.save_pairs(pairs_results, data_reader.test_df, target_pairs)
                results_summary['tasks_run'].append('pairs')

                if 'metrics' in pairs_results:
                    metrics_df = pd.DataFrame(pairs_results['metrics']).T
                    results_summary['pairs_metrics'] = {
                        'avg_mae': float(metrics_df['MAE'].mean()) if 'MAE' in metrics_df.columns else None,
                        'avg_mse': float(metrics_df['MSE'].mean()) if 'MSE' in metrics_df.columns else None,
                        'avg_r2': float(metrics_df['R2'].mean()) if 'R2' in metrics_df.columns else None,
                        'num_targets': len(target_pairs) * 2
                    }
                print(f"    Pairs task completed")
            else:
                print(f"    No target pairs found")

        if task in ['binary', 'all']:
            print(f"\\nStep 4: Training binary classification model...")
            binary_name = data_reader.get_binary_target_name()
            if binary_name:
                classifier = XGBoostClassifier(config)
                binary_results = classifier.train(data_reader, descriptors)
                saver.save_binary(binary_results, data_reader.test_df, binary_name)
                results_summary['tasks_run'].append('binary')

                if 'metrics' in binary_results:
                    results_summary['binary_metrics'] = {
                        'accuracy': float(binary_results['metrics'].get('accuracy', 0)),
                        'precision': float(binary_results['metrics'].get('precision', 0)),
                        'recall': float(binary_results['metrics'].get('recall', 0)),
                        'f1': float(binary_results['metrics'].get('f1', 0)),
                        'roc_auc': float(binary_results['metrics'].get('roc_auc', 0))
                    }
                print(f"    Binary task completed")
            else:
                print(f"    No binary target found")

        if task in ['binary-pairs', 'all']:
            print(f"\\nStep 5: Training binary-pairs classification models...")
            binary_f_cols = data_reader.get_binary_f_columns()
            if binary_f_cols:
                bp_trainer = BinaryPairsTrainer(config)
                bp_results = bp_trainer.train(data_reader, descriptors)
                saver.save_binary_pairs(bp_results, data_reader.test_df, binary_f_cols)
                results_summary['tasks_run'].append('binary_pairs')

                if 'metrics' in bp_results:
                    metrics_df = pd.DataFrame(bp_results['metrics']).T
                    results_summary['binary_pairs_metrics'] = {
                        'avg_accuracy': float(metrics_df['accuracy'].mean()),
                        'avg_precision': float(metrics_df['precision'].mean()),
                        'avg_recall': float(metrics_df['recall'].mean()),
                        'avg_f1': float(metrics_df['f1'].mean()),
                        'avg_roc_auc': float(metrics_df['roc_auc'].mean()),
                        'num_classifiers': len(binary_f_cols)
                    }
                print(f"    Binary-pairs task completed")
            else:
                print(f"    No binary F columns found")

        if task in ['multilabel', 'all']:
            print(f"\\nStep 6: Training multilabel bucket models...")
            bucket_columns = data_reader.get_bucket_columns()
            if bucket_columns:
                multilabel_trainer = MultilabelTrainer(config)
                multilabel_results = multilabel_trainer.train(data_reader, descriptors)
                saver.save_multilabel(multilabel_results, data_reader.test_df, bucket_columns)
                results_summary['tasks_run'].append('multilabel')

                if 'metrics' in multilabel_results:
                    metrics_df = pd.DataFrame(multilabel_results['metrics']).T
                    results_summary['multilabel_metrics'] = {
                        'avg_accuracy': float(metrics_df['accuracy'].mean()),
                        'avg_precision': float(metrics_df['precision'].mean()),
                        'avg_recall': float(metrics_df['recall'].mean()),
                        'avg_f1': float(metrics_df['f1'].mean()),
                        'avg_roc_auc': float(metrics_df['roc_auc'].mean()),
                        'num_buckets': len(bucket_columns)
                    }
                print(f"    Multilabel task completed")
            else:
                print(f"    No bucket columns found")

        print(f"\\n{'='*80}")
        print(f"Experiment Summary")
        print(f"{'='*80}")
        print(f"Tasks run: {', '.join(results_summary['tasks_run'])}")

        return results_summary

    def run_all(self):
        """Run all experiments in the grid."""
        experiments = self.generate_experiment_configs()

        print(f"\\n{'#'*80}")
        print(f"STARTING BENCHMARK: {len(experiments)} experiments")
        print(f"{'#'*80}")

        for i, exp_config in enumerate(experiments, 1):
            print(f"\\nProgress: [{i}/{len(experiments)}] ({100*i/len(experiments):.1f}%)")
            self.run_single_experiment(exp_config)
            self._save_intermediate_summary()

        self._save_final_summary()

        print(f"\\n{'#'*80}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'#'*80}")
        print(f"Completed: {len(self.summary['completed'])}")
        print(f"Failed: {len(self.summary['failed'])}")
        print(f"Skipped: {len(self.summary['skipped'])}")
        print(f"Results directory: {self.run_dir}")

    def _save_intermediate_summary(self):
        """Save intermediate summary after each experiment."""
        summary_path = os.path.join(self.run_dir, 'benchmark_summary_intermediate.json')
        with open(summary_path, 'w') as f:
            json.dump(self._make_summary_serializable(), f, indent=2)

    def _save_final_summary(self):
        """Save final summary with all results."""
        summary_path = os.path.join(self.run_dir, 'benchmark_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self._make_summary_serializable(), f, indent=2)

        if self.summary['completed']:
            csv_data = []
            for exp in self.summary['completed']:
                row = {
                    'experiment_id': exp['id'],
                    'output_dir': exp['output_dir'],
                    'status': 'completed'
                }

                results = exp.get('results', {})
                row['task'] = results.get('task', '')
                row['model_type'] = results.get('model_type', '')
                row['descriptor_type'] = results.get('descriptor_type', '')
                row['random_state'] = results.get('random_state', 42)
                row['train_size'] = results.get('datasets', {}).get('train_size', 0)
                row['val_size'] = results.get('datasets', {}).get('val_size', 0)
                row['test_size'] = results.get('datasets', {}).get('test_size', 0)

                for metric_type in ['pairs_metrics', 'binary_metrics', 'binary_pairs_metrics', 'multilabel_metrics']:
                    if metric_type in results:
                        for key, value in results[metric_type].items():
                            if isinstance(value, (int, float)):
                                row[f'{metric_type}_{key}'] = value

                csv_data.append(row)

            csv_path = os.path.join(self.run_dir, 'benchmark_summary.csv')
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)
            print(f"\\nSummary CSV saved to: {csv_path}")

        print(f"Summary JSON saved to: {summary_path}")

    def _make_summary_serializable(self) -> Dict[str, Any]:
        """Convert summary to JSON-serializable format."""
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        return convert(self.summary)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark experiments for molecular property prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  PYTHONPATH=. python xgboost_training/benchmark.py --config xgboost_training/config.yaml

  # Pass Hydra overrides for GraphLevelDataModule
  PYTHONPATH=. python xgboost_training/benchmark.py \\
      --config xgboost_training/config.yaml \\
      dataset.block_3_split_mode=train \\
      training.random_seed=42
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file with experiments defined')

    args, hydra_overrides = parser.parse_known_args()
    hydra_overrides = [o for o in hydra_overrides if o]

    print("="*80)
    print("BENCHMARK STARTUP")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Raw Hydra overrides: {hydra_overrides}")

    setup_logging()
    runner = BenchmarkRunner(args.config, hydra_overrides)
    runner.run_all()


if __name__ == '__main__':
    main()
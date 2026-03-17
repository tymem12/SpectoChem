import argparse
import warnings
import logging
from rdkit import RDLogger
import numpy as np
from pathlib import Path
import sys

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
    warnings.filterwarnings('ignore')
    logging.getLogger('rdkit').setLevel(logging.CRITICAL)
    RDLogger.DisableLog('rdApp.*')


def run_experiment(config):
    """Run a single experiment with the given config."""
    setup_logging()

    print(f"Task mode: {config.task}")
    print(f"Descriptor: {config.descriptor['type']}")
    print(f"Hydra config dir: {config.hydra_config_dir}")

    task_to_dataset = {
        'binary': 'dataset=TMQM_SPECTO_BINARY',
        'pairs': 'dataset=TMQM_SPECTO_PAIRS',
        'binary-pairs': 'dataset=TMQM_SPECTO_PAIRS',
        'multilabel': 'dataset=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL',
    }
    if hasattr(config, 'task') and config.task in task_to_dataset:
        dataset_override = task_to_dataset[config.task]
        if dataset_override not in config.hydra_overrides:
            config.hydra_overrides.append(dataset_override)
            print(f"Added dataset override: {dataset_override}")

    print("\nStep 1: Loading and preparing data...")
    data_reader = DatasetReader(config)
    data_dict = data_reader.load_and_prepare()

    if config.task in ['pairs', 'binary-pairs', 'all', 'custom']:
        target_pairs = data_reader.get_target_pairs()
        if target_pairs:
            print(f"\nTarget pairs ({len(target_pairs)} pairs):")
            for i, (osc, lam) in enumerate(target_pairs[:5]):
                print(f"  Pair {i+1}: {osc} & {lam}")
            if len(target_pairs) > 5:
                print(f"  ... and {len(target_pairs)-5} more")

    if config.task in ['binary', 'all', 'custom']:
        binary_name = data_reader.get_binary_target_name()
        if binary_name:
            print(f"\nBinary target: {binary_name}")

    if config.task in ['binary-pairs', 'all', 'custom']:
        binary_f_cols = data_reader.get_binary_f_columns()
        if binary_f_cols:
            print(f"\nBinary F classifiers: {len(binary_f_cols)}")
            print(f"  First 3: {binary_f_cols[:3]}")

    if config.task in ['multilabel', 'all', 'custom']:
        bucket_columns = data_reader.get_bucket_columns()
        if bucket_columns:
            print(f"\nMultilabel buckets: {len(bucket_columns)}")
            print(f"  First 3: {bucket_columns[:3]}")

    print("\nStep 2: Generating descriptors...")
    descriptor_gen = DescriptorFactory.create(config)

    all_atom_types, all_atom_coords = data_reader.get_combined_atom_data()
    descriptor_gen.fit(all_atom_types)
    descriptors = descriptor_gen.transform(all_atom_types, all_atom_coords)
    descriptors = descriptors.astype(np.float32)

    print(f"Descriptors shape: {descriptors.shape}")

    saver = ResultsSaver(config)

    if config.task in ['pairs', 'all'] or (config.task == 'custom' and config.custom_tasks.get('pairs')):
        print("\nStep 3: Training XGBoost pairs regression models...")
        trainer = XGBoostTrainer(config)
        pairs_results = trainer.train(data_reader, descriptors, data_reader.get_target_pairs())
        print("\nSaving pairs results...")
        saver.save_pairs(pairs_results, data_reader.test_df, data_reader.get_target_pairs())

    if config.task in ['binary', 'all'] or (config.task == 'custom' and config.custom_tasks.get('binary')):
        print("\nStep 4: Training XGBoost binary classification model...")
        classifier = XGBoostClassifier(config)
        binary_results = classifier.train(data_reader, descriptors)
        print("\nSaving binary results...")
        saver.save_binary(binary_results, data_reader.test_df, data_reader.get_binary_target_name())

    if config.task in ['binary-pairs', 'all'] or (config.task == 'custom' and config.custom_tasks.get('binary_pairs')):
        print("\nStep 5: Training XGBoost binary-pairs classification models...")
        bp_trainer = BinaryPairsTrainer(config)
        bp_results = bp_trainer.train(data_reader, descriptors)
        print("\nSaving binary-pairs results...")
        saver.save_binary_pairs(bp_results, data_reader.test_df, data_reader.get_binary_f_columns())

    if config.task in ['multilabel', 'all'] or (config.task == 'custom' and config.custom_tasks.get('multilabel')):
        print("\nStep 6: Training XGBoost multilabel bucket models...")
        multilabel_trainer = MultilabelTrainer(config)
        multilabel_results = multilabel_trainer.train(data_reader, descriptors)
        print("\nSaving multilabel results...")
        saver.save_multilabel(multilabel_results, data_reader.test_df, data_reader.get_bucket_columns())

    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Models saved in: {config.output['models_dir']}")
    print(f"Results saved in: {config.output['results_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models on chemical descriptors')
    parser.add_argument('--config', type=str, default='xgboost_training/config.yaml',
                       help='Path to configuration file (relative to main directory)')
    parser.add_argument('--hydra-overrides', nargs='*', default=[],
                       help='Hydra overrides to select experiment (e.g., +exp=TMQM_SPECTO_BINARY). '
                            'Required to match GraphLevelDataModule splits.')
    args = parser.parse_args()

    config = load_config(args.config)

    config.hydra_config_dir = './config'
    task_to_dataset = {
        'binary': 'dataset=TMQM_SPECTO_BINARY',
        'pairs': 'dataset=TMQM_SPECTO_PAIRS',
        'binary-pairs': 'dataset=TMQM_SPECTO_PAIRS',
        'multilabel': 'dataset=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL',
    }
    dataset_override = task_to_dataset.get(config.task, 'dataset=TMQM_SPECTO_BINARY')

    config.hydra_overrides = [
        'model=supervised_graph_level',
        'backbone@model.backbone=schnet',
        dataset_override
    ] + (args.hydra_overrides or [])

    if not args.hydra_overrides:
        print("WARNING: No --hydra-overrides specified. Cannot match GraphLevelDataModule splits.")
        print("Example: --hydra-overrides +exp=TMQM_SPECTO_BINARY")
    else:
        print(f"Using Hydra overrides: {config.hydra_overrides}")

    run_experiment(config)


if __name__ == '__main__':
    main()
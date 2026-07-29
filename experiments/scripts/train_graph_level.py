import os
from pathlib import Path
import time

import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from rich import print

from experiments.training_utils import save_metrics, setup_trainer
from gjepa.config import GraphLevelExperimentConfig, GraphLevelJEPAConfig
from gjepa.datasets import GJEPANodeLevelDataModule
from gjepa.datasets.graph_level import GraphLevelDataModule, GraphLevelJEPASamplingDataModule
from gjepa.datasets.node_level import ClusterDatamodule
from gjepa.utils import import_from_string
from gjepa.utils.config import resolve_config

torch.set_float32_matmul_precision("high")

ONLY_TEST = bool(int(os.getenv("ONLY_TEST", 0)))
SAVE_EMBEDDINGS = bool(int(os.getenv("SAVE_EMBEDDINGS", 0)))

@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    global ONLY_TEST
    
    raw_config = resolve_config(cfg)
    config = GraphLevelExperimentConfig.from_raw_config(raw_config)
    print(config)

    split_mode = config.dataset.block_3_split_mode

    # --- ONLY_TEST CHECK ---
    if split_mode == "3test":
        ONLY_TEST = True
        print("\n" + "="*50)
        print(f"NOTICE: `block_3_split_mode` is {split_mode!r}.")
        print("Programmatically setting global ONLY_TEST = True.")
        print("Will skip training and load best checkpoint from the '345' split run.")
        print("="*50 + "\n")
    elif ONLY_TEST:
        print("\n" + "="*50)
        print("NOTICE: Global ONLY_TEST is manually set to True.")
        print("Will skip training and evaluate existing checkpoint in the current directory.")
        print("="*50 + "\n")

    seed_everything(config.training.random_seed)

    model = import_from_string(config.model.type)(config)
    model_config = config.model

    if model_config.name == "supervised":
        datamodule = GraphLevelDataModule(
            dataset_config=config.dataset,
            batch_size=config.training.batch_size,
            pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
        )
    elif model_config.name == "gjepa":
        assert isinstance(model_config, GraphLevelJEPAConfig)
        assert config.pos_encoding is not None
        if model_config.sampling_method == "random":
            datamodule = GraphLevelJEPASamplingDataModule(
                dataset_config=config.dataset,
                batch_size=config.training.batch_size,
                pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
                context_ratio=model_config.context_ratio,
                target_ratio=model_config.target_ratio,
            )
        else:
            raise ValueError(f"Invalid subgraph_method: {model_config.sampling_method}")
    else:
        raise ValueError(f"Invalid model name: {model_config.name}")

    trainer = setup_trainer(
        config=config.training,
        reload_dataloaders_every_n_epochs=model_config.reload_dataloaders_every_n_epochs,
    )

    # --- CONDITIONAL TRAINING ---
    if not ONLY_TEST:
        start_train_time = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        end_train_time = time.perf_counter()

        train_duration_sec = end_train_time - start_train_time
        train_epochs = trainer.current_epoch
        
        # If we trained, we just use the standard "best" string for testing
        test_ckpt_path = "best"
    else:
        import re
        import shutil
        
        train_duration_sec = 0.0
        train_epochs = 0
        
        exp_dir = str(config.training.experiment_dir)
        split_mode_345 = "345"

        # If in '3test' mode, fetch the checkpoint from the '345' folder
        if split_mode == "3test":
            block_3_format = "block_3_{split_mode}"
            expected_substring = block_3_format.format(split_mode=split_mode)
            target_substring = block_3_format.format(split_mode=split_mode_345)
            
            if expected_substring not in exp_dir:
                raise ValueError(
                    f"Expected '{expected_substring}' in the experiment directory path, "
                    f"but got: '{exp_dir}'. The path formatting has changed!"
                )
            
            search_dir = Path(exp_dir.replace(expected_substring, target_substring))
            print(f"Looking for {split_mode_345!r} checkpoint in: {search_dir}")
        else:
            search_dir = Path(exp_dir)
            print(f"Looking for checkpoint in: {search_dir}")
        
        # 1. Check if "best.ckpt" already exists to avoid redundant searching
        existing_best = list(search_dir.rglob("best.ckpt"))
        if existing_best:
            # If multiple exist, take the most recently modified one
            best_ckpt = max(existing_best, key=lambda p: p.stat().st_mtime)
            test_ckpt_path = str(best_ckpt)
            print(f"Found existing 'best.ckpt'! Skipping search and using: {test_ckpt_path}")
        
        # 2. If no "best.ckpt", search for standard PyTorch Lightning epoch checkpoints
        else:
            ckpt_paths = list(search_dir.rglob("epoch=*.ckpt"))
            if not ckpt_paths:
                raise FileNotFoundError(
                    f"Could not find any checkpoint matching 'epoch=*.ckpt' or 'best.ckpt' in {search_dir}. "
                    f"Did you train the model first?"
                )
            
            print(f"\nFound {len(ckpt_paths)} checkpoints:")
            for p in ckpt_paths:
                print(f"  - {p.name}")
            
            # Hardcoded metric rules
            higher_is_better_metrics = ["AUROC", "F1", "ACC", "ACCURACY", "PRECISION", "RECALL", "R2", "R_SQUARED"]
            lower_is_better_metrics = ["MAE", "MSE", "RMSE", "LOSS", "JSD", "WASSERSTEIN", "SID", "STMSE", "SMSE", "SRMSE", "RAW_MAE", "L1", "L2"]
            
            def get_checkpoint_score(p: Path):
                """
                Extracts the score for sorting. 
                Raises an error if the metric is unparseable or unknown.
                """
                # Extract epoch
                epoch_match = re.search(r'epoch=(\d+)', p.name)
                epoch = int(epoch_match.group(1)) if epoch_match else -1
                    
                # Extract validation metric
                metric_match = re.search(r'val_([a-zA-Z0-9_]+)=([0-9\.e\-]+)', p.name)
                if not metric_match:
                    raise ValueError(
                        f"Could not parse a 'val_...=...' metric from checkpoint filename: '{p.name}'. "
                        "Ensure your ModelCheckpoint callback includes the validation metric in the filename."
                    )
                    
                metric_name = metric_match.group(1).upper()
                metric_val = float(metric_match.group(2))
                
                is_higher = any(m in metric_name for m in higher_is_better_metrics)
                is_lower = any(m in metric_name for m in lower_is_better_metrics)
                
                if not is_higher and not is_lower:
                    raise ValueError(
                        f"Unknown validation metric '{metric_name}' in checkpoint '{p.name}'. "
                        "It is not defined in either 'higher_is_better_metrics' or 'lower_is_better_metrics'. "
                        "Please update the hardcoded lists."
                    )
                
                if is_higher:
                    score = metric_val
                else:
                    # Negate so max() picks the smallest error
                    score = -metric_val
                    
                # Sort order: 1st by validation score, 2nd by highest epoch, 3rd by most recent
                return (score, epoch, p.stat().st_mtime)

            # Let python's max() find the absolute best checkpoint using our logic
            best_ckpt = max(ckpt_paths, key=get_checkpoint_score)
            
            print(f"\n=> Selected best checkpoint: {best_ckpt.name}")
            
            # Save it as "best.ckpt" alongside the original
            best_save_path = best_ckpt.parent / "best.ckpt"
            print(f"=> Saving a copy as: {best_save_path}\n")
            shutil.copy2(best_ckpt, best_save_path)
            
            test_ckpt_path = str(best_save_path)

        datamodule.setup(stage="fit")

    # --- TESTING ---
    seed_everything(config.training.random_seed, workers=True)
    
    start_test_time = time.perf_counter()
    test_metrics, *_ = trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt_path)
    end_test_time = time.perf_counter()

    test_duration_sec = end_test_time - start_test_time

    assert trainer.log_dir is not None
    
    timing_metrics = {
        "debug_train_time_seconds": train_duration_sec,
        "debug_train_epochs": train_epochs,
        "debug_test_time_seconds": test_duration_sec
    }

    test_metrics.update(timing_metrics)

    save_metrics(test_metrics, Path(trainer.log_dir))

    if SAVE_EMBEDDINGS:
        save_embeddings(model, trainer, datamodule, trainer.log_dir)

def save_embeddings(
    model: LightningModule, trainer: Trainer, datamodule: LightningDataModule, save_dir: str
) -> None:
    data_loaders = [
        datamodule.train_inference_dataloader(),  # type: ignore[attr-defined]
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ]
    data_loader_names = ["train", "val", "test"]
    embeddings = trainer.predict(model, data_loaders, ckpt_path="best")
    save_file = Path(save_dir) / "best_ckpt_embeddings.pt"
    embedding_dict = {}

    assert isinstance(embeddings, list)
    for embs, dl_name in zip(embeddings, data_loader_names):
        emb_collated = torch.cat([e for e, _ in embs], dim=0)
        y_collated = torch.cat([y for _, y in embs], dim=0)
        embedding_dict[dl_name] = (emb_collated, y_collated)

    torch.save(embedding_dict, save_file)


if __name__ == "__main__":
    main()

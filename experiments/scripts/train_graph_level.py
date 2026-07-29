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
        train_duration_sec = 0.0
        train_epochs = 0
        
        exp_dir = str(config.training.experiment_dir)

        split_mode_345 = "345"

        # If in '3test' mode, fetch the checkpoint from the '345' folder
        if split_mode == "3test":
            # Dynamically construct the strings based on the format
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
            # If ONLY_TEST was manually set for a normal run, look in current dir
            search_dir = Path(exp_dir)
            print(f"Looking for checkpoint in: {search_dir}")
        
        ckpt_paths = list(search_dir.rglob("epoch=*.ckpt"))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"Could not find any checkpoint matching 'epoch=*.ckpt' in {search_dir}. "
                f"Did you train the model first?"
            )
        
        # Get the most recently modified checkpoint
        test_ckpt_path = str(max(ckpt_paths, key=lambda p: p.stat().st_mtime))
        print(f"Found checkpoint! Will test using: {test_ckpt_path}")

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

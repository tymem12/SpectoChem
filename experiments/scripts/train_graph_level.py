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

SAVE_EMBEDDINGS = bool(int(os.getenv("SAVE_EMBEDDINGS", 0)))


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    config = GraphLevelExperimentConfig.from_raw_config(raw_config)
    print(config)

    seed_everything(config.training.random_seed)

    model = import_from_string(config.model.type)(config)

    model_config = config.model

    if model_config.name == "supervised":
        datamodule = GraphLevelDataModule(
            dataset_config=config.dataset,
            batch_size=config.training.batch_size,
            pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,

            #dataloader_mode=config.model.dataloader_mode,
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

    trainer.fit(model, datamodule=datamodule)


    start_train_time = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    end_train_time = time.perf_counter()

    train_duration_sec = end_train_time - start_train_time
    train_epochs = trainer.current_epoch

    seed_everything(config.training.random_seed, workers=True)
    
    start_test_time = time.perf_counter()
    test_metrics, *_ = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    end_test_time = time.perf_counter()

    test_duration_sec = end_test_time - start_test_time

    assert trainer.log_dir is not None
    
    timing_metrics = {
        "debug_train_time_seconds": train_duration_sec,
        "debug_train_epochs": train_epochs,
        "debug_test_time_seconds": test_duration_sec
    }

    test_metrics.update(timing_metrics)

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

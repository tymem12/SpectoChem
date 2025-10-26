import os
from pathlib import Path

import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from rich import print

from experiments.training_utils import save_metrics, setup_trainer
from gjepa.config import ExperimentConfig, JEPAConfig
from gjepa.datasets import GJEPANodeLevelDataModule, KHopDatamodule
from gjepa.datasets.node_level import ClusterDatamodule
from gjepa.utils import import_from_string
from gjepa.utils.config import resolve_config

torch.set_float32_matmul_precision("high")

SAVE_EMBEDDINGS = bool(int(os.getenv("SAVE_EMBEDDINGS", 0)))


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    config = ExperimentConfig.from_raw_config(raw_config)
    print(config)

    seed_everything(config.training.random_seed)

    model = import_from_string(config.model.type)(config)

    if config.model.name == "supervised":
        datamodule: LightningDataModule = KHopDatamodule(
            dataset_config=config.dataset,
            batch_size=config.training.batch_size,
            dataloader_mode=config.model.dataloader_mode,
            pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
        )
    elif config.model.name == "gjepa":
        assert isinstance(config.model, JEPAConfig)
        assert config.pos_encoding is not None
        if config.model.subgraph_method == "k_hop":
            datamodule = GJEPANodeLevelDataModule(
                dataset_config=config.dataset,
                batch_size=config.training.batch_size,
                dataloader_mode=config.model.dataloader_mode,
                pos_enc_path=config.pos_encoding.file,
                num_targets=config.model.num_targets,
                similarity_matrix_file=config.model.similarity_matrix_file,
                context_target_overlap_strategy=config.model.context_target_overlap_strategy,
            )
        elif config.model.subgraph_method == "cluster":
            datamodule = ClusterDatamodule(
                dataset_config=config.dataset,
                batch_size=config.training.batch_size,
                dataloader_mode=config.model.dataloader_mode,
                pos_enc_path=config.pos_encoding.file,
                num_targets=config.model.num_targets,
            )
        else:
            raise ValueError(f"Invalid subgraph_method: {config.model.subgraph_method}")
    else:
        raise ValueError(f"Invalid model name: {config.model.name}")

    trainer = setup_trainer(
        config=config.training,
        reload_dataloaders_every_n_epochs=config.model.reload_dataloaders_every_n_epochs,
    )

    trainer.fit(model, datamodule=datamodule)

    seed_everything(config.training.random_seed, workers=True)
    test_metrics, *_ = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    assert trainer.log_dir is not None
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

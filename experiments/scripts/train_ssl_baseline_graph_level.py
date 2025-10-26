from pathlib import Path

import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from rich import print

from experiments.training_utils import save_metrics, setup_trainer
from gjepa.datasets.graph_level import GraphLevelDataModule
from gjepa.datasets.node_level import KHopDatamodule, load_graph
from gjepa.models.ssl.base import SSLModelBase
from gjepa.models.ssl.bgrl import BGRLModel, BGRLGraphModelPrePool, BGRLGraphModelPostPool
from gjepa.models.ssl.gbt import GBTModel, GBTGraphModelPrePool, GBTGraphModelPostPool
from gjepa.utils import resolve_config
from gjepa.utils.pyg import create_transform

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    print(raw_config)
    seed_everything(raw_config["training"]["random_seed"], workers=True)
    datamodule: GraphLevelDataModule | None = None
    baseline_name = raw_config.pop("baseline_name")
    raw_config["training"]["experiment_dir"] = Path(raw_config["training"]["experiment_dir"]) / baseline_name
    ssl_model: SSLModelBase

    if baseline_name == "gbt-prepool":
        print("Running GBT prepool model")
        ssl_model = GBTGraphModelPrePool(raw_config)
    elif baseline_name == "gbt-postpool":
        print("Running GBT postpool model")
        ssl_model = GBTGraphModelPostPool(raw_config)
    elif baseline_name == "bgrl-prepool":
        print("Running BGRL prepool model")
        ssl_model = BGRLGraphModelPrePool(raw_config)
    elif baseline_name == "bgrl-postpool":
        print("Running BGRL postpool model")
        ssl_model = BGRLGraphModelPostPool(raw_config)
    else:
        raise ValueError(f"Invalid SSL baseline_name: {baseline_name}")

    config = ssl_model.config
    if datamodule is None:
        datamodule = GraphLevelDataModule(
            dataset_config=config.dataset,
            batch_size=config.training.batch_size,
            pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
            # dataloader_mode=config.model.dataloader_mode,
        )
    trainer = setup_trainer(config=config.training, reload_dataloaders_every_n_epochs=0)
    trainer.fit(ssl_model, datamodule=datamodule)

    seed_everything(config.training.random_seed, workers=True)
    test_metrics, *_ = trainer.test(ssl_model, datamodule=datamodule)
    print(test_metrics)
    assert trainer.log_dir is not None
    save_metrics(test_metrics, Path(trainer.log_dir))



if __name__ == "__main__":
    main()

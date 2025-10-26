from pathlib import Path

import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from rich import print

from experiments.training_utils import save_metrics, setup_trainer
from gjepa.datasets.node_level import KHopDatamodule, load_graph
from gjepa.models.ssl.base import SSLModelBase
from gjepa.models.ssl.bgrl import BGRLModel
from gjepa.models.ssl.gbt import GBTModel
from gjepa.models.ssl.gigamae import GiGaMAE, GiGaMAEConfig, KHopDatamoduleWithStructAndPCAEmbs, pca
from gjepa.utils import resolve_config
from gjepa.utils.pyg import create_transform

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    seed_everything(raw_config["training"]["random_seed"], workers=True)

    datamodule: KHopDatamodule | None = None
    baseline_name = raw_config.pop("baseline_name")
    ssl_model: SSLModelBase
    if baseline_name == "gbt":
        ssl_model = GBTModel(raw_config)
    elif baseline_name == "bgrl":
        ssl_model = BGRLModel(raw_config)
    elif baseline_name == "gigamae":
        gigamae_cfg = GiGaMAE.type_Cfg(**raw_config)  # type: ignore[misc]
        datamodule = _prepare_gigamae_datamodule(gigamae_cfg)
        ssl_model = GiGaMAE(gigamae_cfg)
    else:
        raise ValueError(f"Invalid SSL baseline_name: {baseline_name}")

    config = ssl_model.config
    print(config)

    if datamodule is None:
        datamodule = KHopDatamodule(
            dataset_config=config.dataset,
            batch_size=config.training.batch_size,
            dataloader_mode="transductive",
        )

    trainer = setup_trainer(config=config.training, reload_dataloaders_every_n_epochs=0)

    trainer.fit(ssl_model, datamodule=datamodule)

    seed_everything(config.training.random_seed, workers=True)
    test_metrics, *_ = trainer.test(ssl_model, datamodule=datamodule, ckpt_path="best")

    assert trainer.log_dir is not None
    save_metrics(test_metrics, Path(trainer.log_dir))


def _prepare_gigamae_datamodule(config: GiGaMAEConfig) -> KHopDatamoduleWithStructAndPCAEmbs:
    """Performs PCA preprocessing step of graph features for GiGaMAE model."""
    pca_file = config.feat_pca_emb_path
    pca_file.parent.mkdir(exist_ok=True, parents=True)

    if not pca_file.is_file():
        print("Precomputing features PCA...")
        ds_cfg = config.dataset
        data = load_graph(
            name=ds_cfg.name,
            root_dir=ds_cfg.root_dir,
            transform=create_transform(ds_cfg.transforms),
            pre_transform=create_transform(ds_cfg.pre_transforms),
        )

        if config.feat_pca_ratio == 1.0:
            torch.save(data.x, pca_file)
        else:
            x_proj = pca(data.x, ratio=config.feat_pca_ratio)
            torch.save(x_proj, pca_file)

    print("Finished precomputing features PCA...")

    datamodule = KHopDatamoduleWithStructAndPCAEmbs(
        struct_emb_path=config.struct_emb_path,
        pca_emb_path=config.feat_pca_emb_path,
        dataset_config=config.dataset,
        batch_size=config.training.batch_size,
        dataloader_mode="transductive",
        pos_enc_path=None,
    )
    datamodule.setup("")

    config.struct_emb_dim = datamodule.struct_emb_dim
    config.feat_pca_emb_dim = datamodule.feat_pca_dim

    return datamodule


if __name__ == "__main__":
    main()

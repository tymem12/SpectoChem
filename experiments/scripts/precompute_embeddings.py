import hydra
import torch
from omegaconf import DictConfig
from rich import print
from pathlib import Path

from gjepa.config import GraphLevelPrecomputedEmbeddingsConfig
from gjepa.datasets.graph_level import GraphLevelDataModule
from gjepa.utils import import_from_string
from gjepa.utils.config import resolve_config

from gjepa.utils.precomputed_embeddings import generate_embeddings

torch.set_float32_matmul_precision("high")

@hydra.main(version_base="1.3", config_path="../../config", config_name="precompute_embeddings")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    config = GraphLevelPrecomputedEmbeddingsConfig.model_validate(raw_config)
    print(config)

    backbone_config = config.backbone.copy()
    backbone_module_path = backbone_config.pop("gnn_cls")

    encoder = import_from_string(backbone_module_path)(**backbone_config)

    ds_config = config.dataset

    datamodule = GraphLevelDataModule(
        dataset_config=ds_config,
        batch_size=config.batch_size,
        pos_enc_path=config.pos_encoding.file if config.pos_encoding else None
    )

    datamodule.setup(stage="predict")

    encoder_name = backbone_module_path.rsplit(".", maxsplit=1)[-1].removesuffix("Encoder")

    output_subdir = Path(
        ds_config.name,
        ds_config.additional_loading_params["prediction_type"],
        encoder_name
    )

    pool = config.pool

    output_dir = config.output_dir / output_subdir

    metadata = config.model_dump(
        mode="json"
    )

    generate_embeddings(
        datamodule, encoder, pool, output_dir, metadata
    )

if __name__ == "__main__":
    main()

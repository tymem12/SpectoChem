import hydra
import torch
from omegaconf import DictConfig
from rich import print
from pathlib import Path

from gjepa.config import GraphLevelPrecomputeEmbeddingsConfig
from gjepa.datasets.graph_level import GraphLevelDataModule
from gjepa.utils import import_from_string
from gjepa.utils.config import resolve_config

from gjepa.utils.precomputed_embeddings import generate_embeddings

torch.set_float32_matmul_precision("high")

@hydra.main(version_base="1.3", config_path="../../config", config_name="precompute_embeddings")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    config: GraphLevelPrecomputeEmbeddingsConfig = GraphLevelPrecomputeEmbeddingsConfig.from_raw_config(raw_config)
    print(config)

    model = import_from_string(config.model.type)(config)

    ds_config = config.dataset
    model_config = config.model

    if model_config.name == "supervised":
        datamodule = GraphLevelDataModule(
            dataset_config=ds_config,
            batch_size=config.batch_size,
            pos_enc_path=config.pos_encoding.file if config.pos_encoding else None
        )
    else:
        raise ValueError(f"Invalid model name: {model_config.name}")

    encoder_module_path: str = model_config.backbone["gnn_cls"]
    encoder_name = encoder_module_path.rsplit(".", maxsplit=1)[-1].removesuffix("Encoder")

    output_subpath = Path(
        ds_config.name,
        ds_config.additional_loading_params["prediction_type"],
        encoder_name
    )

    output_path = config.output_dir / output_subpath

    metadata = config.model_dump(
        mode="json"
    )

    generate_embeddings(
        datamodule, model, output_path, metadata
    )

if __name__ == "__main__":
    main()

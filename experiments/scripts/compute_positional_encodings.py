from pathlib import Path
import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel

from gjepa.config import DatasetConfig, PosEncodingConfig
from gjepa.datasets.graph_level import load_graph
from gjepa.utils import resolve_config
from gjepa.utils.pyg import create_transform
from gjepa.utils.pos_encoding import save_pe_from_transformed_dataset

import os
class Config(BaseModel, extra="forbid"):
    dataset: DatasetConfig
    pos_encoding: PosEncodingConfig
    random_seed: int


@hydra.main(version_base="1.3", config_path="../../config", config_name="pos_encoding")
def main(cfg: DictConfig) -> None:
    resolved_config = resolve_config(cfg)
    dataset_config = resolved_config['dataset']
    pos_encoding_config = resolved_config['pos_encoding']
    seed_everything(resolved_config['random_seed'])
    
    dataset = load_graph(
        root_dir=Path(dataset_config["root_dir"]),
        name=dataset_config["name"],
        transform=create_transform(dataset_config["transforms"] | pos_encoding_config["transforms"]),
        pre_transform=create_transform(dataset_config["pre_transforms"]),
        additional_loading_params=dataset_config['additional_loading_params'],

    )
    os.makedirs(Path(pos_encoding_config["file"]).parent, exist_ok=True)
    save_pe_from_transformed_dataset(dataset, pos_encoding_config["file"])

if __name__ == "__main__":
    main()

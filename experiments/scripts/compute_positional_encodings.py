import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel

from gjepa.config import DatasetConfig, PosEncodingConfig
from gjepa.datasets.node_level import load_graph
from gjepa.utils import resolve_config
from gjepa.utils.pyg import create_transform


class Config(BaseModel, extra="forbid"):
    dataset: DatasetConfig
    pos_encoding: PosEncodingConfig
    random_seed: int


@hydra.main(version_base="1.3", config_path="../../config", config_name="pos_encoding")
def main(cfg: DictConfig) -> None:
    config = Config(**resolve_config(cfg))
    seed_everything(config.random_seed)

    data = load_graph(
        root_dir=config.dataset.root_dir,
        name=config.dataset.name,
        transform=create_transform(config.dataset.transforms | config.pos_encoding.transforms),
        pre_transform=create_transform(config.dataset.pre_transforms),
    )

    config.pos_encoding.file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.pos, config.pos_encoding.file)


if __name__ == "__main__":
    main()

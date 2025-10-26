from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_and_resolve_config(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return resolve_config(cfg)


def resolve_config(config: DictConfig, resolve: bool = True) -> dict:
    config_primitive = OmegaConf.to_container(config, resolve=resolve)
    assert isinstance(config_primitive, dict)
    return config_primitive

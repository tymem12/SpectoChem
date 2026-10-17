from omegaconf import DictConfig, OmegaConf

def resolve_config(config: DictConfig, resolve: bool = True) -> dict:
    config_primitive = OmegaConf.to_container(config, resolve=resolve)
    assert isinstance(config_primitive, dict)
    return config_primitive

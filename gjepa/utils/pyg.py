from typing import Type

from torch_geometric.transforms import BaseTransform, Compose

from gjepa.utils import import_from_string


def create_transform(transforms: dict[str, dict]) -> BaseTransform:
    return Compose(
        [
            _get_transform_class(trf_name)(**trf_config)
            for trf_name, trf_config in transforms.items()
        ]
    )


def _get_transform_class(name: str) -> Type[BaseTransform]:
    try:
        cls = import_from_string(f"torch_geometric.transforms.{name}")
    except ImportError:
        cls = import_from_string(f"gjepa.transforms.{name}")

    return cls

from typing import Any

from torch_geometric.loader.dataloader import Collater


class EmptyGraphFilteringCollater(Collater):
    def __init__(self) -> None:
        super().__init__(
            dataset=None,  # type: ignore
            follow_batch=None,
            exclude_keys=None,
        )

    def __call__(self, batch: list[dict | None]) -> Any:
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            return None

        return super().__call__(batch)

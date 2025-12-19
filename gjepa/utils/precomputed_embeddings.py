import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import (
    global_add_pool,
    global_max_pool,
    global_mean_pool
)

from typing import Literal, Optional

from gjepa.models.encoders import GNNEncoder
from gjepa.datasets.graph_level import GraphLevelDataModule
from experiments.training_utils import DEVICE

_EMBEDDINGS_FILE_NAME = "raw.pt"

def generate_embeddings(
    data_module: GraphLevelDataModule,
    encoder: GNNEncoder,
    pool: Literal["mean", "max", "sum"],
    output_dir: Path,
    metadata: Optional[dict[str]] = None
):
    encoder.to(DEVICE)
    encoder.eval()

    if metadata is None:
        metadata = {}

    save_dict = {"metadata": metadata}

    split_configs = [
        ("train", data_module.train_ds),
        ("val", data_module.val_ds),
        ("test", data_module.test_ds)
    ]

    match pool:
        case "max":
            pool_fn = global_max_pool
        case "mean":
            pool_fn = global_mean_pool
        case "sum":
            pool_fn = global_add_pool
        case _:
            raise ValueError(f"Invalid pool method {pool!r}")

    print("Starting encoder inference...")

    with torch.no_grad():
        for split_name, dataset_subset in split_configs:
            if dataset_subset is None or len(dataset_subset) == 0:
                raise ValueError(f"Empty split {split_name!r}")

            loader = DataLoader(
                dataset_subset,
                batch_size=data_module.batch_size,
                shuffle=False,
                drop_last=False
            )

            split_embeddings = []
            split_ids = []

            for batch in tqdm(loader, desc=f"Generating {split_name!r} split embeddings"):
                batch = batch.to(DEVICE)

                z = encoder(batch)
                z = pool_fn(z, batch.batch)

                split_embeddings.append(z.cpu())

                if hasattr(batch, "CSD_code"):
                    split_ids.extend(batch.CSD_code)
                else:
                    raise KeyError(f"Batch in {split_name!r} split missing 'CSD_code' attribute.")

            split_tensor = torch.cat(split_embeddings, dim=0)

            if len(split_ids) != split_tensor.shape[0]:
                raise RuntimeError(
                    f"Mismatch in {split_name!r}: {len(split_ids)} "
                    f"IDs vs {split_tensor.shape[0]} embeddings."
                )

            save_dict[split_name] = {
                "embeddings": split_tensor,
                "ids": split_ids
            }

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / _EMBEDDINGS_FILE_NAME

    print("Saving...")

    torch.save(save_dict, output_path)

    metadata_file_path = output_dir / "metadata.json"

    with metadata_file_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=3)

    print(f"Saved embeddings to {output_path.as_posix()!r}")

class PrecomputedEmbeddings:
    def __init__(self, dir_path: Path, device: str = "cpu"):
        emb_path = dir_path / _EMBEDDINGS_FILE_NAME

        print(f"Loading embeddings from {emb_path.as_posix()!r}...")

        data = torch.load(emb_path, map_location=device, weights_only=False)

        self.metadata = data.pop("metadata", {})
        self.data_by_split: dict[str, dict[str, torch.Tensor | list[str]]] = data

        self._id_to_loc: dict[str, tuple[str, int]] = {}

        for split_name, content in self.data_by_split.items():
            for i, csd in enumerate(content["ids"]):
                self._id_to_loc[csd] = (split_name, i)

    def get_embedding(self, csd_code: str) -> torch.Tensor:
        split, idx = self._id_to_loc[csd_code]
        return self.data_by_split[split]["embeddings"][idx]

    def get_split(self, csd_code: str) -> str:
        split, _ = self._id_to_loc[csd_code]
        return split

import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch_geometric.loader import DataLoader

from typing import Optional

from gjepa.models.encoders import GNNEncoder
from gjepa.datasets.graph_level import GraphLevelDataModule
from experiments.training_utils import DEVICE

def generate_embeddings(
    data_module: GraphLevelDataModule,
    encoder: GNNEncoder,
    output_dir: Path,
    metadata: Optional[dict[str]] = None
):
    encoder.to(DEVICE)
    encoder.eval()

    all_embeddings = []
    all_ids = []
    all_splits = []

    split_configs = [
        ("train", data_module.train_ds),
        ("val", data_module.val_ds),
        ("test", data_module.test_ds)
    ]

    print("Starting Inference...")

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

            for batch in tqdm(loader, desc=f"Generating {split_name!r} split embeddings"):
                batch = batch.to(DEVICE)

                out = encoder(batch)

                all_embeddings.append(out.cpu())

                if hasattr(batch, "CSD_code"):
                    all_ids.extend(batch.CSD_code)
                else:
                    raise KeyError(f"Batch in {split_name} split missing 'CSD_code' attribute.")

                all_splits.extend([split_name] * out.size(0))

    final_embeddings = torch.cat(all_embeddings, dim=0)

    if len(all_ids) != final_embeddings.shape[0]:
        raise RuntimeError(f"Mismatch: {len(all_ids)} IDs vs {final_embeddings.shape[0]} embeddings.")

    if metadata is None:
        metadata = {}

    save_dict = {
        "embeddings": final_embeddings,
        "ids": all_ids,
        "splits": all_splits,
        "generated_at": pd.Timestamp.now().isoformat(),
        "metadata": metadata 
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "embeddings.pt"

    print("Saving...")

    torch.save(save_dict, output_path)

    metadata_file_path = output_dir / "metadata.json"

    with metadata_file_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=3)

    print(f"Saved {final_embeddings.shape[0]} embeddings to {output_path.as_posix()!r}")

class PrecomputedEmbeddings:
    def __init__(self, path: str, device: str = "cpu"):
        print(f"Loading embeddings from {path}...")
        data = torch.load(path, map_location=device, weights_only=False)

        self.embeddings = data["embeddings"]
        self.ids = data["ids"]
        self.splits = data["splits"]
        self.metadata = data.get("metadata", {})

        self._id_to_idx = {csd: i for i, csd in enumerate(self.ids)}

    def get_embedding(self, csd_code: str) -> torch.Tensor:
        idx = self._id_to_idx[csd_code]
        return self.embeddings[idx]

    def get_split(self, csd_code: str) -> str:
        idx = self._id_to_idx[csd_code]
        return self.splits[idx]

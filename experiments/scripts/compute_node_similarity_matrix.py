from pathlib import Path
from typing import Literal, Optional

import torch
import typer
from lightning_fabric import seed_everything
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GCN
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

from gjepa.config import DatasetConfig
from gjepa.datasets.node_level import load_graph
from gjepa.utils.config import load_and_resolve_config
from gjepa.utils.pyg import create_transform

T_feature = Literal["node_attribute", "random_gcn", "pos_enc"]
T_similarity = Literal["euclidean", "cosine"]


def main(
    seed: int = typer.Option(...),
    dataset_config: Path = typer.Option(...),
    feature_name: str = typer.Option(...),
    similarity_measure: Optional[str] = typer.Option(None),
    pos_enc_file: Optional[Path] = typer.Option(None),
    output_file: Path = typer.Option(...),
) -> None:
    seed_everything(seed)
    config = DatasetConfig(**load_and_resolve_config(dataset_config))
    data = load_graph(
        root_dir=config.root_dir,
        name=config.name,
        transform=create_transform(config.transforms),
        pre_transform=create_transform(config.pre_transforms),
    )

    if feature_name == "uniform":
        similarity_matrix = torch.ones(data.num_nodes, data.num_nodes)
    elif feature_name == "label":
        similarity_matrix = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float)
        same_label_idx = data.y.unsqueeze(dim=-1) == data.y.flatten().repeat(data.num_nodes, 1)
        similarity_matrix[same_label_idx] = 1
    else:
        train_node_features = extract_node_features(
            graph=data,
            feature_name=feature_name,  # type: ignore
            pos_enc_file=pos_enc_file,
            config=config,
        )

        similarity_matrix = compute_similarity_matrix(
            features=train_node_features,
            similarity_measure=similarity_measure,  # type: ignore
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj=similarity_matrix, f=output_file)


def extract_node_features(
    graph: Data,
    feature_name: T_feature,
    pos_enc_file: Path | None,
    config: DatasetConfig,
) -> Tensor:
    if feature_name == "node_attribute":
        return graph.x

    elif feature_name == "random_gcn":
        gcn = GCN(
            in_channels=graph.num_node_features,
            hidden_channels=128,  # TODO: parameterize
            num_layers=config.num_hops,
        )
        gcn.reset_parameters()
        gcn.eval()

        loader = NeighborLoader(
            data=graph,
            num_neighbors=config.num_neighbors,
            input_nodes=None,
            shuffle=False,
            batch_size=32,  # TODO: parametrize
        )
        features: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                z = gcn(x=batch.x, edge_index=batch.edge_index)[: batch.batch_size]
                features.append(z)
        return torch.cat(features, dim=0)

    elif feature_name == "pos_enc":
        assert pos_enc_file is not None
        pos_encodings = torch.load(pos_enc_file)
        return pos_encodings

    else:
        raise ValueError(f"Invalid feature_name: '{feature_name}'")


def compute_similarity_matrix(
    features: Tensor,
    similarity_measure: T_similarity,
) -> Tensor:
    if similarity_measure == "euclidean":
        sm = pairwise_euclidean_distance(x=features)
        sm = sm / sm.amax(dim=-1, keepdim=True)  # Squash distance to [0, 1]
        sm = 1 - sm  # Now, similar objects will get higher scores
    elif similarity_measure == "cosine":
        sm = pairwise_cosine_similarity(x=features)
        sm = (sm + 1) / 2
    else:
        raise ValueError(f"Invalid similarity_measure: '{similarity_measure}'")

    return sm


if __name__ == "__main__":
    typer.run(main)

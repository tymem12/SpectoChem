import json
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, Type, TypeVar

import hydra
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gjepa.config import DatasetConfig, GraphDatasetConfig
from gjepa.datasets.node_level import KHopDatamodule, GraphDataModule
from gjepa.models.downstream import DownstreamModel, LinearProbingClassifier, LinearProbingRegressor
from gjepa.models.encoders import GNNEncoder
from gjepa.utils import resolve_config

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


T = TypeVar("T", bound=GraphDatasetConfig)


class Config(BaseModel, Generic[T], extra="forbid"):
    dataset: T
    backbone: dict[str, Any]
    random_seed: int
    experiment_dir: Path
    batch_size: int

    @property
    def version(self) -> str:
        return f"version_{self.random_seed}"


class PredictionsSelector(Protocol):
    def __call__(self, batch: Data, z: Tensor) -> tuple[Tensor, Tensor]: ...


@torch.no_grad()
def _compute_reprs(
    model: nn.Module,
    data_loader: DataLoader,
    predictions_selector: PredictionsSelector,
    desc: str = "",
) -> tuple[Tensor, Tensor]:
    model = model.to(DEVICE)
    model.eval()
    zs = []
    ys = []
    for batch in tqdm(data_loader, desc=desc):
        batch = batch.to(DEVICE)
        z = model(batch)
        z, y = predictions_selector(batch, z)
        zs.append(z)
        ys.append(y)

    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def evaluate_model(
    model: nn.Module,
    datamodule: GraphDataModule,
    downstream_model: DownstreamModel,
    predictions_selector: PredictionsSelector,
) -> dict[str, float]:
    z_train, y_train = _compute_reprs(
        model, datamodule.train_dataloader(), predictions_selector, "train"
    )
    downstream_model.update_train(z_train, y_train)

    z_test, y_test = _compute_reprs(
        model, datamodule.test_dataloader(), predictions_selector, "test"
    )
    downstream_model.update_test(z_test, y_test)

    downstream_model.fit()
    metrics = downstream_model.score("test_")

    return {m_name: m_val.tolist() for m_name, m_val in metrics.items()}


def _get_downstream_model(config: Config) -> DownstreamModel:
    dataset_config = config.dataset
    task_type = dataset_config.task_type

    if task_type.endswith("regression"):
        model_cls = LinearProbingRegressor
    else:
        model_cls = LinearProbingClassifier

    model = model_cls(task_type, dataset_config.out_channels)

    return model


def evaluate_model_and_save_results(
    cfg: DictConfig,
    dataset_cfg_cls: Type[T],
    data_module_factory: Callable[[Config[T]], GraphDataModule],
    predictions_selector: PredictionsSelector,
):
    raw_config = resolve_config(cfg)
    config = Config[dataset_cfg_cls](**raw_config)

    seed_everything(config.random_seed, workers=True)

    rand_encoder = GNNEncoder(**config.backbone)

    datamodule = data_module_factory(config)

    datamodule.setup("")

    downstream_model = _get_downstream_model(config)

    test_results = evaluate_model(rand_encoder, datamodule, downstream_model, predictions_selector)

    config.experiment_dir.mkdir(parents=True)
    with (config.experiment_dir / "metrics.json").open("w") as file:
        json.dump(test_results, file, indent="\t")


@hydra.main(version_base="1.3", config_path="../../config/exp_rand_init")
def main(cfg: DictConfig) -> None:
    evaluate_model_and_save_results(
        cfg,
        dataset_cfg_cls=DatasetConfig,
        data_module_factory=lambda config: KHopDatamodule(
            dataset_config=config.dataset,
            batch_size=config.batch_size,
            dataloader_mode="transductive",
        ),
        predictions_selector=lambda batch, z: (z[: batch.batch_size], batch.y[: batch.batch_size]),
    )


if __name__ == "__main__":
    main()

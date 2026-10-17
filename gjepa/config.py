from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Type, TypeVar, Generic

from pydantic import BaseModel, Extra, SerializeAsAny, model_validator

TaskType = Literal["binary", "multiclass", "regression", "multiregression", 'multilabel']
MetricMode = Literal["min", "max"]


class GraphDatasetConfig(BaseModel, extra=Extra.forbid):
    name: str
    root_dir: Path
    in_channels: int
    out_channels: int
    task_type: TaskType
    main_metric: str
    metric_mode: MetricMode
    pre_transforms: dict[str, Any]
    transforms: dict[str, Any]  # dataset-specific transforms


class DatasetConfig(GraphDatasetConfig):
    num_hops: int
    num_neighbors: list[int]
    num_partitions: int | None  # used when subgraph_method = cluster

    @model_validator(mode="after")
    def check_num_neighbors_has_same_len_as_num_hops(self) -> "DatasetConfig":
        if len(self.num_neighbors) != self.num_hops:
            raise ValueError(
                f"len(num_neighbors) ({len(self.num_neighbors)}) "
                f"should be equal num_hops ({self.num_hops})"
            )
        return self


class GraphLevelDatasetConfig(GraphDatasetConfig):
    split_ratios: tuple[float, float] | None
    additional_loading_params: dict[str, Any] | None = None
    target_standarization: bool | None = None
    block_3_split_mode: Optional[Literal["train", "test", "3test", "345"]] = None
    group_by_isomers: bool = False
    predictor_kwargs: Optional[dict[str, Any]] = None


class TrainingConfig(BaseModel, extra=Extra.forbid):
    random_seed: int
    experiment_name: str
    experiment_dir: Path
    learning_rate: float
    weight_decay: float
    batch_size: int
    min_epochs: int | None
    max_epochs: int
    use_tensorboard: bool
    use_wandb: bool
    wandb_project: str | None
    early_stopping: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    scheduler_config: dict[str, Any] | None
    save_representations: bool = False

class GraphModelConfig(BaseModel, extra="forbid"):
    name: str
    type: str
    backbone: dict[str, Any]
    transforms: dict[str, Any]  # model-specific transforms
    reload_dataloaders_every_n_epochs: int

class GraphLevelModelConfig(GraphModelConfig):
    pass

T = TypeVar("T", bound=GraphDatasetConfig)
U = TypeVar("U", bound="GraphExperimentConfig")
V = TypeVar("V", bound=GraphModelConfig)
W = TypeVar("W", bound=str)

class GraphExperimentConfig(BaseModel, Generic[T, V], extra="forbid"):
    dataset: T
    training: TrainingConfig
    model: SerializeAsAny[V]

    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, Type[V]]]

    @classmethod
    def from_raw_config(cls: Type[U], raw_config: dict[str, Any]) -> U:
        model_type = raw_config["model"]["type"]

        model_config_cls = cls.model_cls_path_to_config_cls_mapping[model_type]

        model_cfg = model_config_cls(**raw_config.pop("model"))

        return cls(**raw_config, model=model_cfg)

class ExperimentConfig(GraphExperimentConfig[DatasetConfig, GraphModelConfig]):
    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, GraphModelConfig]] = {
        "gjepa.models.SupervisedNodeLevelGNN": GraphModelConfig,
    }

class GraphLevelExperimentConfig(GraphExperimentConfig[GraphLevelDatasetConfig, GraphLevelModelConfig]):
    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, GraphLevelModelConfig]] = {
        "gjepa.models.SupervisedGraphLevelGNN": GraphLevelModelConfig,
    }

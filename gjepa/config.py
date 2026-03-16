from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Type, TypeVar, Generic

from pydantic import BaseModel, Extra, SerializeAsAny, model_validator

TaskType = Literal["binary", "multiclass", "regression", "multiregression", 'multilabel']
MetricMode = Literal["min", "max"]
T_extractor = Literal["node", "subgraph"]
T_gjepa_predictor = Literal["mlp", "transformer"]
T_gjepa_overlap_strategy = Literal["ignore", "remove", "mask"]
T_dataloader_mode = Literal["inductive", "transductive"]
T_subgraph_method = Literal["k_hop", "cluster"]

T_sampling_method = Literal["random"]
T_graph_level_extractor = Literal["mean_pool", "sum_pool", "max_pool"]

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
    block_3_split_mode: Optional[Literal["train", "test"]] = None
    group_by_isomers: bool = False


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


class PosEncodingConfig(BaseModel, extra="forbid"):
    name: str
    file: Path
    dim: int
    transforms: dict[str, Any]

class GraphModelConfig(BaseModel, extra="forbid"):
    name: str
    type: str
    backbone: dict[str, Any]
    transforms: dict[str, Any]  # model-specific transforms
    reload_dataloaders_every_n_epochs: int

class ModelConfig(GraphModelConfig):
    @property
    def dataloader_mode(self) -> T_dataloader_mode:
        if self.name == "supervised":
            return "inductive"
        return "transductive"

class GraphLevelModelConfig(GraphModelConfig):
    pass

T = TypeVar("T", bound=GraphDatasetConfig)
U = TypeVar("U", bound="GraphExperimentConfig")
V = TypeVar("V", bound=GraphModelConfig)
W = TypeVar("W", bound=str)

class GraphJEPAConfig(BaseModel, Generic[W], extra="forbid"):
    target_predictor: dict[str, Any]
    ema: list[float]
    ipe: float
    ipe_scale: float
    context_extractor_type: W
    target_extractor_type: W

class JEPAConfig(GraphJEPAConfig[T_extractor], ModelConfig, extra="forbid"):
    subgraph_method: T_subgraph_method
    context_target_overlap_strategy: T_gjepa_overlap_strategy
    similarity_matrix_file: str | None
    num_targets: int

    @model_validator(mode="after")
    def check_using_transformer_with_subgraphs(self) -> "JEPAConfig":
        if (
            self.context_extractor_type == "subgraph" or self.target_extractor_type == "subgraph"
        ) and self.target_predictor["name"] != "transformer":
            raise ValueError("You must use transformer with subgraph context/target!")
        return self

class GraphLevelJEPAConfig(GraphJEPAConfig[T_graph_level_extractor], GraphLevelModelConfig, extra="forbid"):
    target_ratio: float
    context_ratio: float
    sampling_method: T_sampling_method

class GraphExperimentConfig(BaseModel, Generic[T, V], extra="forbid"):
    dataset: T
    training: TrainingConfig
    pos_encoding: PosEncodingConfig | None
    model: SerializeAsAny[V]

    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, Type[V]]]

    @classmethod
    def from_raw_config(cls: Type[U], raw_config: dict[str, Any]) -> U:
        model_type = raw_config["model"]["type"]

        model_config_cls = cls.model_cls_path_to_config_cls_mapping[model_type]

        model_cfg = model_config_cls(**raw_config.pop("model"))

        pos_enc_cfg = raw_config.pop("pos_encoding", None)

        return cls(**raw_config, model=model_cfg, pos_encoding=pos_enc_cfg)

class ExperimentConfig(GraphExperimentConfig[DatasetConfig, ModelConfig]):
    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, ModelConfig]] = {
        "gjepa.models.SupervisedNodeLevelGNN": ModelConfig,
        "gjepa.models.GJEPANodeModel": JEPAConfig
    }

class GraphLevelExperimentConfig(GraphExperimentConfig[GraphLevelDatasetConfig, GraphLevelModelConfig]):
    model_cls_path_to_config_cls_mapping: ClassVar[dict[str, GraphLevelModelConfig]] = {
        "gjepa.models.SupervisedGraphLevelGNN": GraphLevelModelConfig,
        "gjepa.models.GJEPAGraphLevelModel": GraphLevelJEPAConfig
    }

class GraphLevelPrecomputedEmbeddingsConfig(BaseModel):
    dataset: GraphLevelDatasetConfig
    backbone: dict[str, Any]
    pos_encoding: PosEncodingConfig | None = None

    batch_size: int
    output_dir: Path
    pool: Literal["mean", "max", "sum"] | None

    output_model_subdir: Path | None = None

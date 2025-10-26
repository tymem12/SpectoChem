import hydra
from omegaconf import DictConfig

from torch_geometric.nn import global_mean_pool

from gjepa.config import GraphLevelDatasetConfig
from gjepa.datasets.graph_level import GraphLevelDataModule

from eval_rand_init import evaluate_model_and_save_results


@hydra.main(version_base="1.3", config_path="../../config/exp_rand_init")
def main(cfg: DictConfig) -> None:
    evaluate_model_and_save_results(
        cfg,
        dataset_cfg_cls=GraphLevelDatasetConfig,
        data_module_factory=lambda config: GraphLevelDataModule(
            dataset_config=config.dataset, batch_size=config.batch_size
        ),
        predictions_selector=lambda batch, z: (global_mean_pool(z, batch.batch), batch.y),
    )


if __name__ == "__main__":
    main()

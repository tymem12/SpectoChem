from typing import Literal

from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics import Metric

from typing import Optional, Dict
import torch
from torchmetrics import Metric
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score


class StandardizedMAE(Metric):
    full_state_update = False

    def __init__(self, y_std: torch.Tensor):
        super().__init__()
        self.register_buffer("y_std", y_std.float())
        self.add_state("sum_abs_err", default=torch.zeros_like(y_std), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        abs_err = (preds - target).abs().sum(dim=0)  
        self.sum_abs_err += abs_err
        self.total += preds.shape[0]

    def compute(self):
        mae_per_target = self.sum_abs_err / self.total
        mae_per_target_std = mae_per_target / self.y_std
        return mae_per_target_std.mean()


def get_default_regression_metrics(
    task_type: Literal["regression", "multiregression"], output_dim: int,
    y_std: torch.Tensor | None = None, **kwargs
) -> MetricCollection:
    """Provides metrics suitable for regression tasks (univariate or multivariate)."""
    if task_type == "regression":
        if output_dim != 1:
            raise ValueError("Univariate regression expects `output_dim`=1")

        metrics = _get_univariate_regression_metrics()
    elif task_type == "multiregression":
        if output_dim < 2:
            raise ValueError("Multivariate regression expects `output_dim` >= 2")

        metrics = _get_multivariate_regression_metrics(output_dim, y_std=y_std, **kwargs)
    else:
        raise ValueError(f"Invalid `task_type` for regression: {task_type!r}")

    return metrics


def _get_univariate_regression_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "MSE": MeanSquaredError(),
            "MAE": MeanAbsoluteError(),
            "R2": R2Score(),
        }
    )



class ReduceMeanWrapper(Metric):
    def __init__(self, base_metric: Metric):
        super().__init__()
        self.base_metric = base_metric

    def update(self, *args, **kwargs):
        self.base_metric.update(*args, **kwargs)

    def compute(self):
        return self.base_metric.compute().mean()

    def reset(self):
        self.base_metric.reset()


class IndexOutputWrapper(Metric):
    """Wybiera jeden wymiar (target_idx) i liczy metrykę tylko dla niego."""
    def __init__(self, base_metric: Metric, target_idx: int):
        super().__init__()
        self.base_metric = base_metric
        self.target_idx = target_idx

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_i = preds[..., self.target_idx]
        target_i = target[..., self.target_idx]
        self.base_metric.update(preds_i, target_i)

    def compute(self):
        return self.base_metric.compute()

    def reset(self):
        self.base_metric.reset()




def _get_multivariate_regression_metrics(
    output_dim: int,
    reduce_mean: bool = False,
    y_std: Optional[torch.Tensor] = None
) -> MetricCollection:

    mean_metrics: Dict[str, Metric] = {
        "MSE": MeanSquaredError(num_outputs=output_dim),
        "MAE": MeanAbsoluteError(num_outputs=output_dim),
        "R2": R2Score(),
    }

    if reduce_mean:
        for name, metric in list(mean_metrics.items()):
            mean_metrics[name] = ReduceMeanWrapper(metric)

    if y_std is not None:
        mean_metrics["std_MAE"] = StandardizedMAE(y_std)

    per_target_metrics: Dict[str, Metric] = {}
    if reduce_mean:
        for i in range(output_dim):
            per_target_metrics[f"MSE_target_{i}"] = IndexOutputWrapper(
                MeanSquaredError(), i
            )
            per_target_metrics[f"MAE_target_{i}"] = IndexOutputWrapper(
                MeanAbsoluteError(), i
            )
            per_target_metrics[f"R2_target_{i}"] = IndexOutputWrapper(
                R2Score(), i
            )

    return MetricCollection({**mean_metrics, **per_target_metrics})
from abc import ABC
from typing import Literal, Optional

from torch import Tensor, nn
import torch
from gjepa.config import TaskType
from gjepa.metrics.regression import get_default_regression_metrics
from gjepa.models.predictors import PredictorBase
from gjepa.models.predictors.losses import get_regression_loss

class RegressorBase(PredictorBase, ABC):
    def __init__(
        self,
        out_channels: int,
        task_type: Literal["regression", "multiregression"],
        y_std=None,
        prediction_type: Optional[Literal["pairs", "vector", 'only_lambdas']] = None,
        spectral_loss: Optional[str] = None,
        **loss_kwargs
    ):
        if prediction_type:
            assert task_type in ["multiregression", 'regression']

        if spectral_loss:
            assert prediction_type == "vector"

        if task_type == "multiregression":
            metrics_kwargs = dict(
                reduce_mean=True
            )
        else:
            metrics_kwargs = {}

        metrics = get_default_regression_metrics(task_type, out_channels, y_std, prediction_type, **metrics_kwargs)
        loss = get_regression_loss(task_type, spectral_loss, **loss_kwargs)
        super().__init__(loss, metrics)

        self.out_channels = out_channels

    def predict(self, x: Tensor) -> Tensor:
        return self(x)

    def logits_to_proba(self, x: Tensor) -> Tensor:
        return x


class LinearRegressor(RegressorBase):
    def __init__(
        self,
        in_channels: int, out_channels: int,
        task_type: TaskType,
        y_std=None,
        prediction_type: Optional[Literal["pairs", "vector", 'only_lambdas']] = None,
        spectral_loss: Optional[str] = None,
        **loss_kwargs
    ):
        super().__init__(out_channels, task_type, y_std, prediction_type, spectral_loss, **loss_kwargs)
        self.linear = nn.Linear(in_channels, out_channels)


    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

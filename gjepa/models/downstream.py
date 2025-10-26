from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Literal, TypeVar

import torch

from torch import Tensor

from torchmetrics import MetricCollection

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from gjepa.metrics.regression import get_default_regression_metrics
from gjepa.metrics.classification import get_default_classification_metrics


class DownstreamModel(ABC):
    @abstractmethod
    def fit(self) -> None:
        """Trains the downstream model."""

    @abstractmethod
    def score(self, metric_prefix: str = "") -> dict[str, Tensor]:
        """Scores downstream model, returns metrics."""


T = TypeVar("T")


class ProbingModel(Generic[T], DownstreamModel):
    def __init__(
        self,
        model: BaseEstimator,
        task_type: T,
        out_channels: int,
        default_metrics_factory: Callable[[T], MetricCollection],
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.out_channels = out_channels

        self._z_train: list[Tensor] = []
        self._y_train: list[Tensor] = []
        self._z_test: list[Tensor] = []
        self._y_test: list[Tensor] = []

        self._default_metrics_factory = default_metrics_factory

    @property
    def z_train(self) -> Tensor:
        return torch.cat(self._z_train, dim=0)

    @property
    def y_train(self) -> Tensor:
        return torch.cat(self._y_train, dim=0)

    @property
    def z_test(self) -> Tensor:
        return torch.cat(self._z_test, dim=0)

    @property
    def y_test(self) -> Tensor:
        return torch.cat(self._y_test, dim=0)

    def update_train(self, z: Tensor, y: Tensor) -> None:
        self._z_train.append(z.detach().cpu())
        self._y_train.append(y.detach().cpu())

    def update_test(self, z: Tensor, y: Tensor) -> None:
        self._z_test.append(z.detach().cpu())
        self._y_test.append(y.detach().cpu())

    def reset(self) -> None:
        self._z_train = []
        self._y_train = []
        self._z_test = []
        self._y_test = []

    @abstractmethod
    def predict(self, z: Tensor) -> Tensor:
        pass

    def fit(self) -> None:
        self.model.fit(self.z_train, self.y_train)

    def score(self, metric_prefix: str = "") -> dict[str, Tensor]:
        y_pred = self.predict(self.z_test)

        metrics = self._default_metrics_factory(self.task_type, self.out_channels).clone(
            prefix=metric_prefix
        )

        metric_vals = metrics(y_pred, self.y_test)

        return metric_vals


class LinearProbingClassifier(ProbingModel):
    def __init__(
        self, task_type: Literal["binary", "multiclass"], out_channels: int, **kwargs: Any
    ) -> None:
        model = LogisticRegression(solver="liblinear", **kwargs)

        if task_type == "multiclass":
            model = OneVsRestClassifier(model)

        super().__init__(
            model,
            task_type,
            out_channels,
            get_default_classification_metrics,
        )

    def predict(self, z: Tensor) -> Tensor:
        y_score = torch.tensor(self.model.predict_proba(z))
        return y_score


class LinearProbingRegressor(ProbingModel):
    def __init__(
        self, task_type: Literal["regression", "multiregression"], out_channels: int, **kwargs: Any
    ) -> None:
        super().__init__(
            LinearRegression(**kwargs), task_type, out_channels, get_default_regression_metrics
        )

    def predict(self, z: Tensor) -> Tensor:
        y_pred = torch.tensor(self.model.predict(z))
        return y_pred

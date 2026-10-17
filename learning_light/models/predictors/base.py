from abc import ABC, abstractmethod

from torch import Tensor, nn
from torchmetrics import MetricCollection


class PredictorBase(nn.Module, ABC):
    """Base class for all predictors modeling p(y|z)."""

    def __init__(self, loss_func: nn.Module, metrics: MetricCollection):
        super().__init__()
        self.loss_func = loss_func
        self.metrics = metrics

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forwards input tensor and returns LOGITS of the model."""

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Forwards input tensor and returns PREDICTIONS of the model."""

    @abstractmethod
    def logits_to_proba(self, x: Tensor) -> Tensor:
        """Transforms logits into predictions."""

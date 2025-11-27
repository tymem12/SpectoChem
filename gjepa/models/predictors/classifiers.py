from abc import ABC
from typing import Literal

from torch import Tensor, nn

from gjepa.config import TaskType
from gjepa.metrics.classification import get_default_classification_metrics
from gjepa.models.predictors import PredictorBase
from gjepa.models.predictors.losses import get_classification_loss


class ClassifierBase(PredictorBase, ABC):
    def __init__(self, out_channels: int, task_type: Literal["binary", "multiclass"]):
        metrics = get_default_classification_metrics(task_type, out_channels)
        loss = get_classification_loss(task_type)
        super().__init__(loss, metrics)

        self.out_channels = out_channels

    def predict(self, x: Tensor) -> Tensor:
        return self.logits_to_proba(self(x))

    def logits_to_proba(self, x: Tensor) -> Tensor:
        if self.out_channels == 1:
            assert x.shape[1] == 1
            return x.sigmoid()
        else:
            assert x.shape[1] > 1
            return x.softmax(dim=1)


class LinearClassifier(ClassifierBase):
    def __init__(self, in_channels: int, out_channels: int, task_type: TaskType):
        super().__init__(out_channels, task_type)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

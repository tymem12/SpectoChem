from abc import ABC
from typing import Literal, Optional

from torch import Tensor

from gjepa.config import TaskType
from gjepa.metrics.classification import get_default_classification_metrics
from gjepa.models.predictors import PredictorBase, MLPPredictorMixin
from gjepa.models.predictors.losses import get_classification_loss


class ClassifierBase(PredictorBase, ABC):
    def __init__(self, out_channels: int, task_type: Literal["binary", "multiclass"]):
        metrics = get_default_classification_metrics(task_type, out_channels)
        loss = get_classification_loss(task_type)
        self.task_type = task_type
        super().__init__(loss, metrics)

        self.out_channels = out_channels

    def predict(self, x: Tensor) -> Tensor:
        return self.logits_to_proba(self(x))

    def logits_to_proba(self, x: Tensor) -> Tensor:
        if self.task_type in ("binary", "multilabel", "binary_multitask"):
            return x.sigmoid()

        elif self.task_type == "multiclass":
            assert x.shape[1] == self.out_channels
            return x.softmax(dim=1)

        raise ValueError(f"Unsupported task_type in logits_to_proba: {self.task_type}")

class MLPClassifier(MLPPredictorMixin, ClassifierBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_type: TaskType,
        hidden_channels: Optional[list[int]] = None,
        activation: Optional[str] = None
    ):
        ClassifierBase.__init__(
            self, out_channels, task_type
        )

        MLPPredictorMixin.__init__(
            self, in_channels, out_channels, hidden_channels, activation
        )

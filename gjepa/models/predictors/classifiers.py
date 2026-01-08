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

class LinearClassifier(ClassifierBase):
    def __init__(self, in_channels: int, out_channels: int, task_type: TaskType):
        super().__init__(out_channels, task_type)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

# MORE ADVANCED PREDICTION HEAD TODO- TRY SOMETHING BETTER 
# class LinearClassifier(ClassifierBase):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         task_type: TaskType,
#         hidden: int | None = None,
#         dropout: float = 0.1,
#     ):
#         super().__init__(out_channels, task_type)
#         hidden = hidden or in_channels

#         self.pre = nn.Sequential(
#             nn.LayerNorm(in_channels),
#             nn.Linear(in_channels, hidden),
#             nn.GELU(),
#         )
#         self.block = nn.Sequential(
#             nn.LayerNorm(hidden),
#             nn.Linear(hidden, hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
#         self.out = nn.Linear(hidden, out_channels)

#         self.skip = nn.Identity() if hidden == hidden else nn.Identity()

#     def forward(self, x: Tensor) -> Tensor:
#         h = self.pre(x)
#         h = h + self.block(h)
#         return self.out(h)

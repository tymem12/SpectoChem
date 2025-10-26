from typing import Literal

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss


def get_classification_loss(
    task_type: Literal["binary", "binary_multitask", "multiclass"],
) -> nn.Module:
    if task_type == "binary":
        return BCEWithLogitsLoss()
    elif task_type == "multiclass":
        return CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid task_type for classification loss: {task_type}")


def get_regression_loss(task: Literal["regression", "multiregression"]) -> nn.Module:
    if task in ("regression", "multiregression"):
        return MSELoss()
    else:
        raise ValueError(f"Invalid task_type for regression loss: {task}")

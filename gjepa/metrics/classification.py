from typing import Literal

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


def get_default_classification_metrics(
    task_type: Literal["binary", "binary_multitask", "multiclass", "multilabel"],
    output_dim: int,
) -> MetricCollection:
    """Provides metrics suitable for the output dimension (binary/multiclass/multilabel classification)."""
    if task_type == "binary":
        return _get_binary_classification_metrics(output_dim)
    elif task_type == "multiclass":
        return _get_multiclass_classification_metrics(output_dim)
    elif task_type == "multilabel":
        return _get_multilabel_classification_metrics(output_dim)
    else:
        raise ValueError("Invalid task type for classification")


def _get_binary_classification_metrics(output_dim: int) -> MetricCollection:
    if output_dim > 1:
        raise ValueError("MultiOutput not supported yet!")

    return MetricCollection(
        {
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
            "F1": BinaryF1Score(),
            "AUROC": BinaryAUROC(),
            "Accuracy": BinaryAccuracy(),
        }
    )


def _get_multiclass_classification_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "Precision": MulticlassPrecision(num_classes, average="macro"),
            "Recall": MulticlassRecall(num_classes, average="macro"),
            "F1": MulticlassF1Score(num_classes, average="macro"),
            "AUROC": MulticlassAUROC(num_classes, average="macro"),
            "Accuracy": MulticlassAccuracy(num_classes=num_classes),
        }
    )


def _get_multilabel_classification_metrics(num_labels: int) -> MetricCollection:
    """Multilabel classification where multiple labels/nodes can be active per sample."""
    return MetricCollection(
        {
            "Precision": MultilabelPrecision(num_labels=num_labels, average="macro"),
            "Recall": MultilabelRecall(num_labels=num_labels, average="macro"),
            "F1": MultilabelF1Score(num_labels=num_labels, average="macro"),
            "AUROC": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "Accuracy": MultilabelAccuracy(num_labels=num_labels),
        }
    )

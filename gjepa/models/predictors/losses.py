from typing import Callable, Literal, Optional

from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss

from gjepa.utils.spectral_loss import sid, jsd, smse, wasserstein

class SpectralLoss(nn.Module):
    """
    Wrap spectral loss functions (sid, jsd, smse, wasserstein) for training.
    The loss_type string is matched (case-insensitive) against the function names.
    """
    _func_list: list[Callable[[Tensor, Tensor], Tensor]] = [sid, jsd, smse, wasserstein]

    def __init__(self, loss_type: str, **kwargs):
        """
        Args:
            loss_type: Name of the spectral loss function to use (case-insensitive)
            kwargs: Additional keyword arguments passed to the spectral function
        """
        super().__init__()
        loss_type_lower = loss_type.lower()

        # Find function whose name matches loss_type
        matched_func = None
        for f in self._func_list:
            if f.__name__.lower() == loss_type_lower:
                matched_func = f
                break

        if matched_func is None:
            available = [f.__name__ for f in self._func_list]
            raise ValueError(f"Unknown spectral loss '{loss_type}'. Available: {available}")

        self.func = matched_func
        self.loss_type = self.func.__name__  # store canonical name
        self.kwargs = kwargs

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            preds: predicted spectra (batch_size x bins)
            target: ground-truth spectra (batch_size x bins)
        Returns:
            scalar loss
        """
        loss_vals = self.func(preds, target, **self.kwargs)  # per-sample loss
        return loss_vals.mean()  # reduce over batch

def get_classification_loss(
    task_type: Literal["binary", "binary_multitask", "multiclass", "multilabel"],
) -> nn.Module:
    if task_type == "binary":
        return BCEWithLogitsLoss()
    elif task_type == "multiclass":
        return CrossEntropyLoss()
    elif task_type == "multilabel":
        return BCEWithLogitsLoss()
    else:
        raise ValueError(f"Invalid task_type for classification loss: {task_type}")


def get_regression_loss(
    task: Literal["regression", "multiregression"],
    spectral: Optional[str] = None,
    **kwargs
) -> nn.Module:
    if spectral:
        assert task == "multiregression"

        return SpectralLoss(spectral, **kwargs)

    if task in ("regression", "multiregression"):
        return MSELoss()

    raise ValueError(f"Invalid task_type for regression loss: {task}")

import copy
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

T_schedule = Literal["cosine", "linear"]


class EMAModel(nn.Module):
    """Teacher model for self-distillation with EMA weights updates."""

    def __init__(self, online_model: nn.Module, schedule: T_schedule, **schedule_kwargs: Any):
        super().__init__()
        self.target = self.init_target(online_model)
        self.momentum_scheduler = ScheduleMomentum.create_scheduler(schedule, **schedule_kwargs)

    @property
    def last_momentum(self) -> float:
        assert self.momentum_scheduler.last_momentum is not None
        return self.momentum_scheduler.last_momentum

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.target(*args, **kwargs)

    @torch.no_grad()
    def update_model(self, online_model: nn.Module, step: int) -> None:
        momentum = self.momentum_scheduler.get_next(step)
        for p, new_p in zip(self.target.parameters(), online_model.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    @staticmethod
    def init_target(model: nn.Module) -> nn.Module:
        target = copy.deepcopy(model)

        for param in target.parameters():
            param.requires_grad = False

        return target


class ScheduleMomentum(ABC):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.last_momentum: float | None = None

    def get_next(self, step: int) -> float:
        momentum = self._next(step)
        self.last_momentum = momentum
        return momentum

    @abstractmethod
    def _next(self, step: int) -> float:
        return NotImplemented

    @classmethod
    def create_scheduler(
        cls, schedule: T_schedule, *args: Any, **kwargs: Any
    ) -> "ScheduleMomentum":
        if schedule == "cosine":
            return CosineScheduleMomentum(*args, **kwargs)
        elif schedule == "linear":
            return LinearScheduleMomentum(*args, **kwargs)
        else:
            raise ValueError(f"Invalid momentum schedule: {schedule}")


class CosineScheduleMomentum(ScheduleMomentum):
    """Cosine schedule of EMA momentum, like in BGRL (https://arxiv.org/abs/2102.06514)."""

    def __init__(self, max_steps: int, temperature: float):
        super().__init__(max_steps)
        self.temperature = temperature

    def _next(self, step: int) -> float:
        return 1 - ((1 - self.temperature) / 2) * (1 + np.cos(step * np.pi / self.max_steps))


class LinearScheduleMomentum(ScheduleMomentum):
    """Linear schedule of EMA momentum, like in IJEPA (https://arxiv.org/abs/2301.08243)."""

    def __init__(self, max_steps: int, momentum_min: float, momentum_max: float):
        super().__init__(max_steps)
        self.mm_min = momentum_min
        self.mm_max = momentum_max

    def _next(self, step: int) -> float:
        return self.mm_min + step * (self.mm_max - self.mm_min) / self.max_steps

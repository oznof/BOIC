from typing import Callable, Union
from numbers import Number

import torch
from gpytorch import Module


def make_getter(k: str) -> Callable[[Module], torch.Tensor]:
    def getter(self: Module) -> torch.Tensor:
        constraint = self._constraints.get(f"{k}_constraint")
        return constraint.transform(getattr(self, k)) if constraint else getattr(self, k)
    return getter


def make_setter(k: str) -> Callable[[Module, Union[Number, torch.Tensor]], None]:
    def setter(self: Module, v: Union[Number, torch.Tensor]):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(getattr(self, k))
        constraint = self._constraints.get(f"{k}_constraint")
        self.initialize(**{k: constraint.inverse_transform(v) if constraint else v})
    return setter


def make_property(k: str) -> property:
    # must be on raw parameters
    if 'raw_' not in k:
        k = 'raw_' + k
    return property(make_getter(k), make_setter(k))

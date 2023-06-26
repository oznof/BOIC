from typing import Callable

import torch
from torch.distributions import Distribution


class TrainingDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, nll: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._nll = nll

    def log_prob(self, targets):
        return -self._nll(targets)
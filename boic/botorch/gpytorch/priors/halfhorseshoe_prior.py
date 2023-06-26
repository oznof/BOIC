import math
from numbers import Number

import torch
from torch.distributions import HalfCauchy, Normal, constraints
from torch.nn import Module as TModule

from gpytorch.priors.prior import Prior


class HalfHorseshoePrior(Prior):

    arg_constraints = {"scale": constraints.positive}
    support = constraints.positive
    _validate_args = True

    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        if isinstance(scale, Number):
            scale = torch.tensor(float(scale))
        self.K = 1 / math.sqrt(2 * math.pi ** 3)
        self.scale = scale
        super().__init__(scale.shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.scale
        self.register_buffer("scale", scale)
        self._transform = transform

    def log_prob(self, X):
        A = (self.scale / self.transform(X)) ** 2
        lb = self.K / 2 * torch.log(1 + 4 * A)
        ub = self.K * torch.log(1 + 2 * A)
        log_prob = torch.log((lb + ub) / 2) + math.log(2)
        log_prob[X.expand(log_prob.shape) <= 0] = -float('inf')
        return log_prob

    def rsample(self, sample_shape=torch.Size([])):
        local_shrinkage = HalfCauchy(1).rsample(self.scale.shape)
        param_sample = torch.abs(Normal(0, local_shrinkage * self.scale).rsample(sample_shape))
        return param_sample

    def expand(self, expand_shape, _instance=None):
        batch_shape = torch.Size(expand_shape)
        return HalfHorseshoePrior(self.scale.expand(batch_shape))
"""The Sparse Axis-Aligned Subspaces (SAAS) hyperprior on lengthscales.

Eriksson and Jankowiak (2021) assume that only a few dimensions affect the objective:
the inverse squared lengthscales follow a half-Cauchy distribution, 1 / lengthscale^2 ~
HC(tau), whose global shrinkage tau pushes most lengthscales to be long. Appendix H of
the paper trains the Gaussian process for several values of tau and keeps the model
with the highest leave-one-out likelihood (see ``ExactGPModel.train``).
"""
import torch
from gpytorch.priors.prior import Prior
from torch.distributions import constraints
from torch.nn import Module as TModule


class SAASPrior(Prior):
    arg_constraints = {"tau": constraints.positive}
    support = constraints.greater_than_eq(0.)
    _validate_args = True

    def __init__(self, tau=0.1, validate_args=False):
        TModule.__init__(self)
        tau = torch.as_tensor(tau, dtype=torch.get_default_dtype())
        self.register_buffer("tau", tau)
        self._validate_args = validate_args
        self._transform = lambda x: torch.pow(1 / x, 2)

    def log_prob(self, x):
        x = self.transform(x)
        return torch.distributions.HalfCauchy(self.tau, validate_args=self._validate_args).log_prob(x)

    def rsample(self, sample_shape=torch.Size([])):
        x = torch.distributions.HalfCauchy(self.tau, validate_args=self._validate_args).rsample(sample_shape)
        x = 1 / torch.sqrt(x)
        return x

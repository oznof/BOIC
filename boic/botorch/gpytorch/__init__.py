from .constraints import Positive
from .kernels import Distance, _covar_dist, RBFKernel, MaternKernel, CylindricalKernel, InformativeKernel
from .priors import HalfHorseshoePrior
from .means import QuadMean
from .mll import ExactMarginalLogLikelihood

import gpytorch

from gpytorch import (
    beta_features,
    constraints,
    distributions,
    lazy,
    likelihoods,
    mlls,
    models,
    optim,
    settings,
    utils,
    variational,
)

from gpytorch.functions import (  # Deprecated
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    log_normal_cdf,
    logdet,
    matmul,
    root_decomposition,
    root_inv_decomposition,
)
from gpytorch.lazy import cat, delazify, lazify
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal


# apply patches
gpytorch.mlls.ExactMarginalLogLikelihood = ExactMarginalLogLikelihood
gpytorch.constraints.constraints.Positive = Positive

kernels = gpytorch.kernels
kernels.kernel.Distance = Distance
kernels.kernel.Kernel.covar_dist = _covar_dist
#kernels.rbf_kernel.RBFKernel = RBFKernel
kernels.RBFKernel = RBFKernel
#kernels.matern_kernel.MaternKernel = MaternKernel
kernels.MaternKernel = MaternKernel
# patch cylindrical kernel
kernels.InformativeKernel = CylindricalKernel
# add new kernel: informative kernel
kernels.InformativeKernel = InformativeKernel
kernels.__all__.append("InformativeKernel")

priors = gpytorch.priors
priors.HalfHorseshoePrior = HalfHorseshoePrior
priors.__all__.append('HalfHorseshoePrior')

means = gpytorch.means
means.QuadMean = QuadMean
means.__all__.append('QuadMean')


__all__ = [
    # Submodules
    "constraints",
    "distributions",
    "kernels",
    "lazy",
    "likelihoods",
    "means",
    "mlls",
    "models",
    "optim",
    "priors",
    "utils",
    "variational",
    # Classes
    "Module",
    "ExactMarginalLogLikelihood",
    "MultivariateNormal",
    # Functions
    "add_diag",
    "add_jitter",
    "cat",
    "delazify",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "settings"
]
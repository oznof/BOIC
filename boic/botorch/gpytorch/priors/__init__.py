import gpytorch
from gpytorch.priors import *

from .halfhorseshoe_prior import HalfHorseshoePrior
from .kumaraswamy_prior import KumaraswamyPrior
# add other priors as needed
# NOTE: these classes are generated dynamically and have additional features compared to the ones above
from .priors import make_prior, LogNormalPrior, SpikeUniformSlabPrior, LogSpikeUniformSlabPrior


__all__ = gpytorch.priors.__all__ + ['HalfHorseshoePrior', 'KumaraswamyPrior',
                                     'LogNormalPrior', 'SpikeUniformSlabPrior',
                                     'LogSpikeUniformSlabPrior']
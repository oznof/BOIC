from typing import Type, Sequence, Optional, Union
import copy
from numbers import Number

import torch
from torch.distributions import biject_to, Distribution, TransformedDistribution, constraints
from torch.nn import Module
from torch.distributions import StudentT, Uniform, Normal, MultivariateNormal, HalfNormal, Gamma, LogNormal, Dirichlet
from gpytorch.priors import Prior

from .constraint import Constraint
from .truncated_normal import TruncatedNormal
from .mixture import Mixture


def _bufferize_attributes(module: Module, attributes: Sequence[str]):
    if not isinstance(module, TransformedDistribution):
        attr_clones = {attr: getattr(module, attr).clone() for attr in attributes}
        for attr, value in attr_clones.items():
            delattr(module, attr)
            module.register_buffer(attr, value)


class _InitializePrior(object):
    def __call__(self, *args):
        obj = _InitializePrior()
        prior = make_prior(*args)
        obj.__class__ = prior.__class__ if isinstance(prior, Distribution) else prior
        return obj


def make_prior(cls_or_inst: Union[Type[Distribution], Distribution],
               initial_value: Optional[Union[torch.Tensor, Number]] = None) -> Prior:
    isinst = isinstance(cls_or_inst, torch.distributions.Distribution)
    cls = cls_or_inst.__class__ if isinst else cls_or_inst
    support = copy.deepcopy(cls_or_inst.support)

    def __init__(self, *args, initial_value=None, transform=None, **kwargs):
        Module.__init__(self)
        cls.__init__(self, *args, **kwargs)
        # _bufferize_attributes(self, params)
        self._transform = transform
        # the following allows to define gpytorch.constraints only using the support of the chosen prior
        self.constraint = Constraint(support=support, initial_value=initial_value)
        # make the reduce op easier by storing arg
        self._cls_or_inst = cls_or_inst

    def __reduce__(self):
        return _InitializePrior(), (self._cls_or_inst, self.constraint.initial_value), self.__dict__

    cls_prior = type(f"{cls.__name__}Prior", (Prior, cls), {'__init__': __init__, '__reduce__': __reduce__})
    cls_prior.support = support
    if isinst:  # create instance
        params = cls_or_inst.__init__.__code__.co_varnames[:cls_or_inst.__init__.__code__.co_argcount]
        params = [p for p in params if p not in ('self', 'validate_args')]
        params_dict = {p: getattr(cls_or_inst, ('base_dist' if p == 'base_distribution' else p)) for p in params}
        cls_or_inst_prior = cls_prior(initial_value=initial_value, **params_dict)
    else:
        cls_or_inst_prior = cls_prior
    return cls_or_inst_prior

NormalPrior = make_prior(Normal)
MultivariateNormalPrior = make_prior(MultivariateNormal)
HalfNormalPrior = make_prior(HalfNormal)
GammaPrior = make_prior(Gamma)
StudentTPrior = make_prior(StudentT)
LogNormalPrior = make_prior(LogNormal)
DirichletPrior = make_prior(Dirichlet)


def SpikeUniformSlabPrior(loc, low, high, spike_scale=0.01, validate_args=None, **kwargs):
    spike = TruncatedNormal(loc=loc, scale=spike_scale, low=low, high=high, validate_args=validate_args)
    slab = Uniform(low=low, high=high, validate_args=validate_args)
    return make_prior(Mixture([spike, slab]), **kwargs)

def SpikeNormalSlabPrior(loc, low, high, spike_scale=0.01, slab_scale=None, validate_args=None, **kwargs):
    if slab_scale is None:
        slab_scale = 3 * spike_scale
    spike = TruncatedNormal(loc=loc, scale=spike_scale, low=low, high=high, validate_args=validate_args)
    slab = TruncatedNormal(loc=loc, scale=slab_scale, low=low, high=high, validate_args=validate_args)
    return make_prior(Mixture([spike, slab]), **kwargs)


class LogSpikeUniformSlabPrior(Prior, TransformedDistribution):
    arg_constraints = {'loc': constraints.dependent, 'low': constraints.dependent,
                       'high': constraints.dependent, 'spike_scale': constraints.positive}
    reparametrized_params = ['loc', 'low', 'high', 'spike_scale']

    def __init__(self, loc=0., low=0., high=1., spike_scale=0.01, validate_args=None, **kwargs):
        spike = TruncatedNormal(loc=loc, scale=spike_scale, low=low, high=high, validate_args=validate_args)
        slab = Uniform(low=low, high=high, validate_args=validate_args)
        base_dist = Mixture([spike, slab])
        #torch.nn.Module.__init__(self)
        self._transform = None
        #self.add_module('base_dist', base_dist)
        super().__init__(base_dist, torch.distributions.ExpTransform(), validate_args=validate_args)

    # @property
    # def base_dist(self):
    #     return self._base_dist
    #
    # @base_dist.setter
    # def base_dist(self, value):
    #     self._base_dist = value

    @property
    def spike(self):
        return self.base_dist.base_dists[0]

    @property
    def slab(self):
        return self.base_dist.base_dists[1]

    @property
    def loc(self):
        return self.spike.loc

    @property
    def low(self):
        return self.spike.low

    @property
    def high(self):
        return self.spike.high

    @property
    def spike_scale(self):
        return self.spike.scale
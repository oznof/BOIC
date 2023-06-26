from typing import Union, Optional
from numbers import Number
import math

import torch
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .utils import normal_prob, normal_log_prob, normal_cdf, inverse_normal_cdf


class TruncatedNormal(Distribution):

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive,
                       'low': constraints.dependent, 'high': constraints.dependent}
    has_rsample = True

    def __init__(self, loc: Union[float, torch.Tensor], scale: Union[float, torch.Tensor],
                 low: Union[float, torch.Tensor], high: Union[float, torch.Tensor],
                 validate_args: Optional[bool] = None):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        self._store_aux_vars()
        if isinstance(loc, Number) and isinstance(scale, Number) and \
           isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("TruncatedNormal is not defined when low>= high")

    def _store_aux_vars(self):
        self._alpha = self.standardize(self.low)
        self._beta = self.standardize(self.high)
        self._phi_alpha, self._phi_beta = normal_prob(self._alpha), normal_prob(self._beta)
        self._psi_alpha, self._psi_beta = normal_cdf(self._alpha), normal_cdf(self._beta)
        self._normalizer = self._psi_beta - self._psi_alpha
        self._entropy_const = torch.log(math.sqrt(2 * math.pi * math.e) * self.scale * self._normalizer)
        self._log_prob_const = -torch.log(self.scale * self._normalizer)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        new._store_aux_vars()
        super(new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)

    def standardize(self, value):
        return (value - self.loc) / self.scale

    @property
    def mean(self):
        return self.loc + self.scale * (self._phi_alpha - self._phi_beta) / self._normalizer

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        term1 = (self._alpha * self._phi_alpha - self._beta * self._phi_beta) / self._normalizer
        term2 = ((self._phi_alpha - self._phi_beta) / self._normalizer).pow(2)
        return self.scale.pow(2) * (1 + term1 - term2)

    def entropy(self):
        return (self._entropy_const + (self._alpha * self._phi_alpha - self._beta * self._phi_beta) /
                (2 * self._normalizer))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = value.ge(self.low).type_as(self.low)
        return lb * (normal_cdf(self.standardize(value)) - self._psi_alpha) / self._normalizer

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * inverse_normal_cdf(self._normalizer * value + self._psi_alpha)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = value.ge(self.low).type_as(self.low)
        ub = value.lt(self.high).type_as(self.low)
        return torch.log(lb.mul(ub)) + normal_log_prob(self.standardize(value)) + self._log_prob_const

    def sample(self, sample_shape=torch.Size(), generator=None):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            if generator:  # rand throws TypeError if generator is None
                rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device, generator=generator)
            else:
                rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.icdf(rand)

    def rsample(self, sample_shape=torch.Size(), generator=None):
        shape = self._extended_shape(sample_shape)
        if generator:  # rand throws TypeError if generator is None
            rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device, generator=generator)
        else:
            rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(rand)

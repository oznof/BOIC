from gpytorch.priors.prior import Prior
import torch
from torch.distributions import Uniform, constraints
from torch.nn import Module as TModule
# from torch.distributions.kumaraswamy import Kumaraswamy

class KumaraswamyPrior(Prior):

    # def __init__(self, *args, transform=None, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._transform = transform

    arg_constraints = {"a": constraints.positive, 'b': constraints.positive}
    support = constraints.half_open_interval(1e-6, 1)
    _validate_args = True

    def __init__(self, a, b, validate_args=False, transform=None):
        TModule.__init__(self)
        a = torch.as_tensor(a, dtype=torch.get_default_dtype())
        b = torch.as_tensor(b, dtype=torch.get_default_dtype())
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self._transform = transform

    def log_prob(self, x):
        a = self.a.to(x)
        b = self.b.to(x)
        logp = torch.log(a * b) + (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x ** a)
        logp[logp != logp] = -float('inf')
        return logp

    def rsample(self, sample_shape=torch.Size([])):
        unif = Uniform(0, 1).rsample(sample_shape)
        param_sample = (1 - (1 - unif) ** (1 / self.b)) ** (1 / self.a)
        return param_sample
from typing import Sequence, Optional
import copy
from collections import OrderedDict

import torch
from torch.distributions import Distribution
from torch.distributions import constraints

class Mixture(Distribution):

    arg_constraints = {'weights': constraints.simplex}

    def __init__(self, base_dists: Sequence[Distribution], weights: Optional[torch.Tensor] = None):
        supp_dict0 = base_dists[0].support.__dict__
        try:
            assert(all([type(b.support) for b in base_dists]))
            for b in base_dists[1:]:
                supp_dict1 = b.support.__dict__
                assert(supp_dict0.keys() == supp_dict1.keys())
                for k in supp_dict0:
                    assert(torch.all(torch.as_tensor(supp_dict0[k] == supp_dict1[k])))
        except AssertionError:
            raise RuntimeError("All base_dists must have the same support.")
        self.base_dists = list(base_dists)
        num_dists = len(base_dists)
        self._support = copy.deepcopy(self.base_dists[0].support)
        self.weights = weights if weights is not None else torch.ones(num_dists) / num_dists
        batch_shape = self.base_dists[0].batch_shape
        event_shape = self.base_dists[0].event_shape
        Distribution.__init__(self, batch_shape=batch_shape, event_shape=event_shape, validate_args=False)

    @property
    def support(self):
        return self._support

    @property
    def has_rsample(self):
        return all([b.has_rsample for b in self.base_dists])

    @property
    def mean(self):
        mean_list = [w * b.mean for w, b in zip(self.weights, self.base_dists)]
        return torch.stack(mean_list).sum(dim=0)

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        term1 = torch.stack([w * torch.pow(b.mean, 2) for w, b in zip(self.weights, self.base_dists)]).sum(dim=0)
        term2 = torch.stack([w * b.variance for w, b in zip(self.weights, self.base_dists)]).sum(dim=0)
        return term1 + term2 - torch.pow(self.mean, 2)

    def cdf(self, value):
        return torch.stack([w * b.cdf(value) for w, b in zip(self.weights, self.base_dists)]).sum(dim=0)

    def log_prob(self, value):
        log_probs = torch.stack([b.log_prob(value) for b in self.base_dists], dim=-1)
        max_log_probs = log_probs.max(dim=-1, keepdim=True)[0]
        max_log_probs[torch.isinf(max_log_probs)] = 0.
        log_probs = max_log_probs.squeeze() + torch.log((self.weights * torch.exp(log_probs -
                                                                                  max_log_probs)).sum(dim=-1))
        log_probs[torch.isnan(log_probs)] = -float('Inf')
        return log_probs

    def sample(self, sample_shape=torch.Size(), generator: Optional[torch.Generator] = None) -> torch.Tensor:
        with torch.no_grad():
            mask_shape = sample_shape + self.batch_shape
            num_samples = int(torch.as_tensor(mask_shape).prod().item())
            mask = self.weights.multinomial(num_samples, replacement=True,
                                            generator=generator).to(torch.long).reshape(mask_shape)
            if generator:  # sample methods may not support generator kwarg
                torch_rng_state = torch.get_rng_state()
                torch.set_rng_state(generator.get_state())
            samples = torch.stack([b.sample(sample_shape) for b in self.base_dists], dim=0)
            if generator:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(torch_rng_state)
            sample_shape = samples[0].shape
            samples = samples.view((self.weights.numel(), -1) + self._event_shape)
            mask = mask.view(-1).expand((1, ) + self.event_shape + (-1,)).transpose(0, -1).squeeze(-1).unsqueeze(0)
            samples = torch.gather(samples, 0, mask)
            samples = samples.reshape(sample_shape)
            return samples

    def rsample(self, sample_shape=torch.Size(), generator: Optional[torch.Generator] = None) -> torch.Tensor:
        mask_shape = sample_shape + self.batch_shape
        num_samples = torch.as_tensor(mask_shape).prod().item()
        mask = self.weights.multinomial(num_samples, replacement=True,
                                        generator=generator).to(torch.long).reshape(mask_shape)
        torch_rng_state = torch.get_rng_state()
        if generator:  # most sample methods do not support generator arg
            torch.set_rng_state(generator.get_state())
        samples = torch.stack([b.rsample(sample_shape) for b in self.base_dists], dim=0)
        if generator:
            generator.set_state(torch.get_rng_state())
            torch.set_rng_state(torch_rng_state)
        sample_shape = samples[0].shape
        samples = samples.view((self.weights.numel(), -1) + self._event_shape)
        mask = mask.view(-1).expand((1,) + self.event_shape + (-1,)).transpose(0, -1).squeeze(-1).unsqueeze(0)
        samples = torch.gather(samples, 0, mask)
        samples = samples.reshape(sample_shape)
        return samples

    def __repr__(self):
        base_dist_names = ', '.join(OrderedDict({b.__class__.__name__: None for b in self.base_dists}).keys())
        return f"{self.__class__.__name__}(weights : {self.weights}, [{base_dist_names}])"
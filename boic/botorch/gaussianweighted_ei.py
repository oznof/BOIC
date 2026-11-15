"""Gaussian-weighted Expected Improvement, the belief-augmented acquisition functions of
Appendix I of the paper.

Following piBO (Hvarfner et al., 2022), Expected Improvement (EI) is multiplied by a prior
over the location of the optimum, raised to a power that decays with the number of
evaluations: EI(x) * pi(x) ** (zeta / t), with zeta = N / 10 for an evaluation budget N.
The prior is a Gaussian with a diagonal covariance, whose standard deviation is 25% of the
width of the search space, centered at the origin (X0) or at the best point found so far (XA).

- GWEI (``normalization=True``): the normalized Gaussian density, as in piBO.
- GKEI (``normalization=False``): the Gaussian kernel, without its normalizing constant.
  In high dimensions, the normalized density makes EI so small that the acquisition
  function becomes hard to optimize; the kernel avoids this.
- ``prob_floor``: a constant added to the prior before the power, so that the weight never
  reaches zero. The ``v0`` variants of the settings use 1e-12. They are not described in
  the paper's text; the 2023 analysis notebooks label the v0 runs as I+XA+GKEIX0 and
  I+XA+GKEIXA, the names in the paper's figures.
"""
from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
try:  # BoTorch 0.6.0, the version pinned in requirements.txt; removed in later versions
    from botorch.acquisition.objective import ScalarizedObjective
except ImportError:  # pragma: no cover
    ScalarizedObjective = object
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.multivariate_normal import _batch_mahalanobis, MultivariateNormal


class UnnormalizedMultivariateNormal(MultivariateNormal):
    """A multivariate normal whose log-density omits the normalizing constant: a Gaussian kernel."""

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        return -0.5 * M


class GaussianWeightedExpectedImprovement(AnalyticAcquisitionFunction):
    def __init__(self, model: Model, best_f: Union[float, Tensor], reference_input: Union[Tensor] = None,
                 beta_t: Union[float, Tensor] = 10, covar: Union[Tensor, None] = None, bounds: Union[Tensor, None] = None,
                 objective: Optional[ScalarizedObjective] = None, maximize: bool = True, prob_floor: Union[float, Tensor] = 0,
                 normalization: bool = True) -> None:
        # objective is passed only when given, so that later BoTorch versions, which removed it, also work
        super().__init__(model=model, **({'objective': objective} if objective is not None else {}))
        train_inputs = model.train_inputs if isinstance(model.train_inputs, torch.Tensor) else model.train_inputs[0]
        self.dim = train_inputs.size(-1)
        self.maximize = maximize
        self.normalization = normalization
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)
        if reference_input is None:
            reference_input = 0
        if not torch.is_tensor(reference_input):
            reference_input = torch.as_tensor(reference_input).detach().clone().reshape(-1) * torch.ones(self.dim)
            assert(reference_input.size(-1) == self.dim)
        self.register_buffer("reference_input", reference_input)
        if not torch.is_tensor(beta_t):
            beta_t = torch.as_tensor(beta_t)
        self.register_buffer("beta_t", beta_t)
        if covar is None:
            if bounds is None:
                bounds = torch.ones(2, self.dim)  # default: [-1, 1]^D
                bounds[0] = bounds[0] * -1
            lb, ub = torch.as_tensor(bounds)
            covar = ((ub - lb) * 0.25) ** 2  # default: standard deviation is 25% of domain
        covar = torch.as_tensor(covar).detach().clone()
        covar = torch.eye(self.dim) * covar
        mvn = MultivariateNormal if self.normalization else UnnormalizedMultivariateNormal
        self.mvn = mvn(reference_input, covariance_matrix=covar)
        if not torch.is_tensor(prob_floor):
            prob_floor = torch.tensor(prob_floor)
        self.register_buffer("prob_floor", prob_floor)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Gaussian-weighted Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `b1 x ... bk`-dim tensor of acquisition values at the given design points `X`.
        """
        best_f = self.best_f.to(X)
        beta_t = self.beta_t.to(X)
        prob_floor = self.prob_floor.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        out = sigma * (updf + u * ucdf)
        out = out * (torch.exp(self.mvn.log_prob(X)).reshape_as(out) + prob_floor) ** beta_t
        return out

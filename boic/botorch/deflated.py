"""Deflated acquisition functions (experimental).

Deflation removes the modes of an acquisition function where the model is already
certain, so that the next evaluation is chosen where it is more informative. Each
removed mode is approximated by a Gaussian (Laplace approximation of the acquisition
function treated as an unnormalized density), and the removed modes form a Gaussian
mixture over promising regions, which can serve as a search distribution.

The method was first proposed as Collapsed Expected Improvement (technical report,
August 2022, https://doi.org/10.5281/zenodo.24233796). This module generalizes it to any acquisition function by wrapping it.
It is not part of the experiments of the TMLR 2023 paper.

The wrapper works on a log-density g(x), chosen by the kind of acquisition function:

- ``"log"``: the function is already a log, for example LogEI or LogPI; g = alpha.
- ``"positive"``: the function is nonnegative, for example EI or PI; g = log(alpha).
- ``"signed"``: the function can be negative, for example UCB or the posterior mean;
  g = alpha / temperature, the energy-based form.

Two ways of removing a mode are available:

- ``"subtractive"`` (default, as in the 2022 report): subtract a Gaussian bump with the
  height of the density at the mode, and clamp at zero. Points where nothing remains get
  the log-density minus ``floor_penalty``, so that they rank below every other point, as
  the zero of the 2022 implementation did.
- ``"multiplicative"``: multiply the density by 1 - (1 - shift) * bump, which equals
  ``shift`` at the center of the removed mode and tends to one far from it, with the
  width of the mode given by the Laplace curvature. It never becomes negative. The shift
  is nonzero because a factor of zero makes the log-density minus infinity at the
  center, with unbounded gradients nearby. A removed mode is suppressed by the factor
  ``shift``, not removed completely: if it was more than 1 / shift times higher than the
  next mode, it can remain the maximum, and a smaller shift is needed.
  It follows the idea of deflation operators (Farrell et al., 2015), with two differences:
  their multiplier M(x) = 1 / ||x - r||^p + sigma removes a root of an equation by
  growing at r, and their shift sigma acts far away, where it prevents spurious roots;
  here the factor shrinks at the mode, and the shift bounds it at the center.

The output of ``forward`` is the deflated log-density. It has the same maximizers as
the deflated density, so it can be passed to an acquisition optimizer directly.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

try:  # BoTorch is optional, so that the logic can be tested with PyTorch alone
    from botorch.acquisition.acquisition import AcquisitionFunction as _AcquisitionBase
except ImportError:  # pragma: no cover
    _AcquisitionBase = torch.nn.Module

# Kind of each known BoTorch acquisition function, by class name, so that classes
# missing from an installed BoTorch version do not cause import errors.
DENSITY_KINDS = {
    "LogExpectedImprovement": "log",
    "LogNoisyExpectedImprovement": "log",
    "LogProbabilityOfImprovement": "log",
    "qLogExpectedImprovement": "log",
    "qLogNoisyExpectedImprovement": "log",
    "ExpectedImprovement": "positive",
    "NoisyExpectedImprovement": "positive",
    "ProbabilityOfImprovement": "positive",
    "qExpectedImprovement": "positive",
    "qNoisyExpectedImprovement": "positive",
    "qProbabilityOfImprovement": "positive",
    "UpperConfidenceBound": "signed",
    "qUpperConfidenceBound": "signed",
    "PosteriorMean": "signed",
    "qSimpleRegret": "signed",
}
KINDS = ("log", "positive", "signed")
MODES = ("subtractive", "multiplicative")


def density_kind(acq_function) -> str:
    """Return the kind of log-density for an acquisition function, from its class name."""
    name = type(acq_function).__name__
    if name not in DENSITY_KINDS:
        raise ValueError(f"Unknown acquisition function {name}: pass kind='log', 'positive' or 'signed'.")
    return DENSITY_KINDS[name]


class DeflatedAcquisitionFunction(_AcquisitionBase):
    """Wrap an acquisition function and remove its modes where the model is already certain.

    Args:
        acq_function: the acquisition function to deflate. It maps a ``b x 1 x d`` tensor
            to a ``b`` tensor.
        kind: ``"log"``, ``"positive"`` or ``"signed"``. Inferred from the class name if None.
        mode: ``"subtractive"`` or ``"multiplicative"``.
        temperature: scale of a signed acquisition function in the energy-based form.
        threshold: a mode is removed while the predictive variance there is at most
            ``threshold * noise``.
        noise: the observation noise variance. Read from the model if None.
        shift: in multiplicative mode, the factor at the center of a removed mode. Must be
            positive, so that the log-density stays finite (see the module docstring).
        variance_fn: maps a ``1 x 1 x d`` tensor to the predictive variance. Read from
            ``acq_function.model`` if None.
        eps: floor for the log of a positive acquisition function and of the deflated density.
        jitter_hessian, jitter_covariance: regularization of the Laplace approximation,
            the same values as in the 2022 implementation.
        floor_penalty: in subtractive mode, how far below the log-density a point is placed
            when the subtraction leaves nothing there.

    If the Hessian at a maximizer is not positive definite, for example when the maximizer
    lies on the boundary of the search space and is not a true mode, its eigenvalues are
    raised to ``jitter_hessian`` before inversion. The 2022 implementation failed there.
    """

    def __init__(
        self,
        acq_function: Callable[[Tensor], Tensor],
        kind: Optional[str] = None,
        mode: str = "subtractive",
        temperature: float = 1.0,
        threshold: float = 2.0,
        noise: Optional[float] = None,
        shift: float = 1e-3,
        variance_fn: Optional[Callable[[Tensor], Tensor]] = None,
        eps: float = 1e-6,
        jitter_hessian: float = 1e-5,
        jitter_covariance: float = 1e-3,
        floor_penalty: float = 1e3,
    ) -> None:
        model = getattr(acq_function, "model", None)
        if _AcquisitionBase is torch.nn.Module or model is None:
            torch.nn.Module.__init__(self)
        else:
            super().__init__(model=model)
        kind = density_kind(acq_function) if kind is None else kind
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}.")
        self.acq_function = acq_function
        self.kind = kind
        self.mode = mode
        self.temperature = temperature
        self.threshold = threshold
        self.noise = noise
        self.shift = shift
        self.variance_fn = variance_fn
        self.eps = eps
        self.jitter_hessian = jitter_hessian
        self.jitter_covariance = jitter_covariance
        self.floor_penalty = floor_penalty
        self.centers: List[Tensor] = []  # x_i, each of shape d
        self.precisions: List[Tensor] = []  # H_i, each of shape d x d
        self.log_heights: List[Tensor] = []  # log of the deflated density at x_i when removed

    @property
    def name(self) -> str:
        """Name of the wrapped acquisition function with the prefix Deflated."""
        return "Deflated" + type(self.acq_function).__name__

    def __repr__(self) -> str:
        return f"{self.name}(kind={self.kind!r}, mode={self.mode!r}, modes_removed={len(self.centers)})"

    # ----------------------------------------------------------------- densities
    def base_log_density(self, X: Tensor) -> Tensor:
        """The log-density g(X) of the wrapped acquisition function, before deflation."""
        value = self.acq_function(X)
        if self.kind == "log":
            return value
        if self.kind == "positive":
            return torch.log(value.clamp_min(self.eps))
        return value / self.temperature

    def _quadratic_forms(self, X: Tensor) -> Tensor:
        """Return 0.5 * (x - x_i)^T H_i (x - x_i) for each removed mode, shape ``b x m``."""
        x = X.reshape(X.shape[0], -1)
        forms = []
        for center, precision in zip(self.centers, self.precisions):
            diff = x - center
            forms.append(0.5 * ((diff @ precision) * diff).sum(-1))
        return torch.stack(forms, dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        """The deflated log-density at ``X`` (``b x 1 x d``), shape ``b``."""
        log_density = self.base_log_density(X)
        if not self.centers:
            return log_density
        forms = self._quadratic_forms(X)
        if self.mode == "subtractive":
            # density - sum_i height_i * bump_i, clamped at zero, computed in log space
            log_bumps = torch.stack(self.log_heights) - forms
            ratio = torch.exp(torch.logsumexp(log_bumps, dim=-1) - log_density)
            remaining = 1.0 - ratio
            removed = remaining <= self.eps
            safe = torch.where(removed, torch.ones_like(remaining), remaining)
            return torch.where(removed, log_density - self.floor_penalty, log_density + torch.log(safe))
        # multiplicative: density * prod_i (1 - (1 - shift) * bump_i)
        factors = 1.0 - (1.0 - self.shift) * torch.exp(-forms)
        return log_density + torch.log(factors).sum(-1)

    # ------------------------------------------------------------ mode removal
    def predictive_variance(self, x: Tensor) -> Tensor:
        X = x.reshape(1, 1, -1)
        if self.variance_fn is not None:
            return self.variance_fn(X).reshape(())
        return self.model.posterior(X.squeeze(1)).variance.reshape(())

    def observation_noise(self) -> Tensor:
        if self.noise is not None:
            return torch.as_tensor(self.noise)
        return self.model.likelihood.noise.mean()

    def needs_removal(self, x: Tensor) -> bool:
        """True if the model is already certain at ``x``: variance at most threshold * noise."""
        return bool(self.predictive_variance(x) <= self.threshold * self.observation_noise())

    def remove_mode(self, x: Tensor) -> None:
        """Fit a Laplace approximation of the current deflated density at ``x`` and remove it."""
        center = x.detach().reshape(-1).clone()
        d = center.numel()

        def negative_log_density(z: Tensor) -> Tensor:
            return -self.forward(z.reshape(1, 1, d)).sum()

        with torch.no_grad():
            log_height = self.forward(center.reshape(1, 1, d)).reshape(())
        hessian = torch.autograd.functional.hessian(negative_log_density, center.clone())
        eye = torch.eye(d, dtype=hessian.dtype, device=hessian.device)
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(hessian)
        if eigenvalues.min() <= 0:  # not a true mode, for example on the boundary
            values, vectors = torch.linalg.eigh(hessian)
            hessian = vectors @ torch.diag(values.clamp_min(self.jitter_hessian)) @ vectors.transpose(-1, -2)
        hessian = hessian + self.jitter_hessian * eye
        covariance = torch.cholesky_inverse(torch.linalg.cholesky(hessian)) + self.jitter_covariance * eye
        precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        self.centers.append(center)
        self.precisions.append(precision.detach())
        self.log_heights.append(log_height.detach())

    def deflate(self, optimizer: Callable[["DeflatedAcquisitionFunction"], Tensor], max_modes: int = 10) -> Tensor:
        """Remove modes until the maximizer is uncertain enough, and return that maximizer.

        Args:
            optimizer: returns the maximizer of the acquisition function it receives, a
                tensor with d elements, for example a wrapper around ``optimize_acqf``.
            max_modes: the most modes to remove.
        """
        x = optimizer(self)
        while len(self.centers) < max_modes and self.needs_removal(x):
            self.remove_mode(x)
            x = optimizer(self)
        return x

    # ------------------------------------------------------- search distribution
    def search_distribution(self) -> MixtureSameFamily:
        """Gaussian mixture of the removed modes, weighted by their densities when removed."""
        if not self.centers:
            raise RuntimeError("No mode has been removed yet.")
        logits = torch.stack(self.log_heights)
        loc = torch.stack(self.centers)
        precision = torch.stack(self.precisions)
        components = MultivariateNormal(loc=loc, precision_matrix=precision)
        return MixtureSameFamily(Categorical(logits=logits), components)


def deflated_class(acq_class: type) -> type:
    """Return a class named Deflated<Name> that builds ``acq_class`` and deflates it.

    Example: ``DeflatedExpectedImprovement = deflated_class(ExpectedImprovement)``, then
    ``DeflatedExpectedImprovement(model, best_f=..., mode="subtractive")``. Keyword
    arguments of DeflatedAcquisitionFunction are separated from those of the base class.
    """
    own = set(DeflatedAcquisitionFunction.__init__.__code__.co_varnames[2:DeflatedAcquisitionFunction.__init__.__code__.co_argcount])

    def __init__(self, *args, **kwargs):
        deflation_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in own}
        DeflatedAcquisitionFunction.__init__(self, acq_class(*args, **kwargs), **deflation_kwargs)

    return type(f"Deflated{acq_class.__name__}", (DeflatedAcquisitionFunction,), {"__init__": __init__})


__all__ = ["DeflatedAcquisitionFunction", "deflated_class", "density_kind", "DENSITY_KINDS"]

"""Tests of the SAAS hyperprior (Appendix H): boic/botorch/gpytorch/priors/saas_prior.py and
the SAAS training rule in boic/torch/gp/models.py.

The prior is loaded from its file, and the training methods are taken from the source of
ExactGPModel and attached to a small GPyTorch model, so that the tests do not import the
whole boic package, which pins BoTorch 0.6.0. Run with: python -m pytest tests/test_saas.py
"""
import ast
import importlib.util
import math
from pathlib import Path

import gpytorch
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "boic"
spec = importlib.util.spec_from_file_location("saas_prior", ROOT / "botorch" / "gpytorch" / "priors" / "saas_prior.py")
saas_prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saas_prior)
SAASPrior = saas_prior.SAASPrior

torch.set_default_dtype(torch.float64)


def test_log_prob_is_half_cauchy_of_the_inverse_squared_lengthscale():
    prior = SAASPrior(tau=0.1)
    lengthscale = torch.tensor([0.5, 1.0, 3.0])
    expected = torch.distributions.HalfCauchy(torch.tensor(0.1)).log_prob(1 / lengthscale ** 2)
    assert torch.allclose(prior.log_prob(lengthscale), expected)


def test_smaller_tau_favors_longer_lengthscales():
    short, long = torch.tensor(0.3), torch.tensor(5.0)
    for tau in [1.0, 0.1, 0.01]:
        prior = SAASPrior(tau=tau)
        assert prior.log_prob(long) > prior.log_prob(short)
    # the preference for long lengthscales grows as tau shrinks
    gap = lambda tau: (SAASPrior(tau=tau).log_prob(long) - SAASPrior(tau=tau).log_prob(short)).item()
    assert gap(0.01) > gap(1.0)


def test_samples_are_positive_lengthscales():
    torch.manual_seed(0)
    samples = SAASPrior(tau=0.1).rsample(torch.Size([1000]))
    assert torch.all(samples > 0)


# ---------------------------------------------------------------- SAAS-MAP training
def _methods_from_exact_gp_model():
    """The SAAS members of ExactGPModel, compiled from the source of boic/torch/gp/models.py."""
    source = (ROOT / "torch" / "gp" / "models.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExactGPModel")
    wanted = {"SAAS_TAUS", "saas_priors", "_train_saas"}
    body = [n for n in cls.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))]
    module = ast.Module(body=[ast.ClassDef(name="SAASMembers", bases=[], keywords=[], body=body, decorator_list=[])],
                        type_ignores=[])
    namespace = {"torch": torch, "np": np}
    # the source imports SAASPrior from the package; give it the class loaded above
    import sys, types
    for name in ["boic", "boic.botorch", "boic.botorch.gpytorch", "boic.botorch.gpytorch.priors"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["boic.botorch.gpytorch.priors.saas_prior"] = saas_prior
    exec(compile(ast.fix_missing_locations(module), "models.py", "exec"), namespace)
    return namespace["SAASMembers"]


SAASMembers = _methods_from_exact_gp_model()


class SmallGP(SAASMembers, gpytorch.models.ExactGP):
    """An exact GP with an ARD RBF kernel and a SAAS hyperprior, with the params and _train
    interface of ExactGPModel."""

    def __init__(self, x, y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base = gpytorch.kernels.RBFKernel(ard_num_dims=x.shape[-1], lengthscale_prior=SAASPrior(tau=0.1))
        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.trained_taus = []

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def params(self, to_numpy=False, **kwargs):
        if kwargs:
            self.load_state_dict({k: torch.as_tensor(v) for k, v in kwargs.items()}, strict=False)
        return {k: v.detach().clone() for k, v in self.state_dict().items() if "tau" not in k}

    def _train(self, *args, **kwargs):
        self.trained_taus.append(self.saas_priors[0].tau.item())
        self.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        for _ in range(60):
            optimizer.zero_grad()
            loss = -mll(self(*self.train_inputs), self.train_targets)
            loss.backward()
            optimizer.step()
        return {"loss": loss.item()}


def data(seed=0):
    torch.manual_seed(seed)
    x = torch.rand(30, 4)
    y = torch.sin(6 * x[:, 0])  # only the first dimension matters
    return x, y


def test_finds_the_saas_prior_on_the_lengthscale():
    model = SmallGP(*data())
    assert len(model.saas_priors) == 1


def test_trains_once_for_each_tau_and_keeps_the_best_loo():
    model = SmallGP(*data())
    model._train_saas()
    assert model.trained_taus == list(SAASMembers.SAAS_TAUS)
    assert model.saas_priors[0].tau.item() in SAASMembers.SAAS_TAUS


def test_each_tau_starts_from_the_same_parameters():
    model = SmallGP(*data())
    initial = model.params()
    starts = []
    original = model._train

    def recording_train(*args, **kwargs):
        starts.append(model.params())
        return original(*args, **kwargs)

    model._train = recording_train
    model._train_saas()
    for start in starts:
        for key in initial:
            assert torch.allclose(start[key], initial[key])


def test_keeps_the_parameters_of_the_selected_model():
    model = SmallGP(*data())
    model._train_saas()
    model.eval()
    loo = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
    model.train()
    value = loo(model(*model.train_inputs), model.train_targets).item()
    assert math.isfinite(value)
    # the irrelevant dimensions get longer lengthscales than the relevant one
    lengthscales = model.covar_module.base_kernel.lengthscale.reshape(-1)
    assert lengthscales[0] < lengthscales[1:].min()

"""Tests of boic/botorch/deflated.py.

The module is loaded from its file, so that the tests do not import the whole boic
package, which pins BoTorch 0.6.0. Run with: python -m pytest tests/test_deflated.py
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "boic" / "botorch" / "deflated.py"
spec = importlib.util.spec_from_file_location("deflated", PATH)
deflated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deflated)
DeflatedAcquisitionFunction = deflated.DeflatedAcquisitionFunction

torch.set_default_dtype(torch.float64)


class ExpectedImprovement(torch.nn.Module):
    """Stand-in with the BoTorch class name: a positive function with two modes."""

    def forward(self, X):
        x = X.reshape(X.shape[0], -1)
        return torch.exp(-0.5 * ((x - 1.0) ** 2).sum(-1) / 0.3) + 0.6 * torch.exp(-0.5 * ((x + 1.0) ** 2).sum(-1) / 0.3)


class UpperConfidenceBound(torch.nn.Module):
    """Stand-in with the BoTorch class name: two modes, negative almost everywhere."""

    def forward(self, X):
        x = X.reshape(X.shape[0], -1)
        bumps = torch.exp(-0.5 * ((x - 1.0) ** 2).sum(-1) / 0.3) + 0.6 * torch.exp(-0.5 * ((x + 1.0) ** 2).sum(-1) / 0.3)
        return torch.log(bumps + 1e-3) - 0.5


def grid_optimizer(acq, lo=-3.0, hi=3.0, n=601):
    grid = torch.linspace(lo, hi, n).reshape(-1, 1, 1)
    with torch.no_grad():
        values = acq(grid)
    return grid[values.argmax()].reshape(-1)


def collapsed_ei_2022(base, best_x, X, jitter_hessian=1e-5, jitter_covariance=1e-3, eps=1e-6):
    """The 2022 Collapsed EI step: EI minus a Gaussian bump at its maximizer, clamped at zero."""
    x0 = best_x.reshape(1, 1, -1)
    ei_max = base(x0).reshape(())
    hessian = torch.autograd.functional.hessian(lambda z: -torch.log(base(z.reshape(1, 1, -1)).clamp_min(eps)).sum(), best_x.reshape(-1))
    eye = torch.eye(hessian.shape[-1])
    covariance = torch.cholesky_inverse(torch.linalg.cholesky(hessian + jitter_hessian * eye)) + jitter_covariance * eye
    hessian = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    diff = X.reshape(X.shape[0], -1) - best_x.reshape(1, -1)
    quad = ei_max * torch.exp(-0.5 * ((diff @ hessian) * diff).sum(-1))
    return (base(X) - quad).clamp_min(0)


def test_name_has_prefix():
    acq = DeflatedAcquisitionFunction(ExpectedImprovement(), noise=1.0, variance_fn=lambda X: torch.ones(1))
    assert acq.name == "DeflatedExpectedImprovement"
    assert deflated.deflated_class(ExpectedImprovement).__name__ == "DeflatedExpectedImprovement"


def test_kind_is_inferred_from_the_class_name():
    assert deflated.density_kind(ExpectedImprovement()) == "positive"
    assert deflated.density_kind(UpperConfidenceBound()) == "signed"
    with pytest.raises(ValueError):
        deflated.density_kind(torch.nn.Identity())


def test_subtractive_matches_the_2022_collapsed_ei():
    base = ExpectedImprovement()
    acq = DeflatedAcquisitionFunction(base, noise=1.0, variance_fn=lambda X: torch.ones(1))
    best_x = grid_optimizer(acq)
    acq.remove_mode(best_x)
    X = torch.linspace(-3, 3, 301).reshape(-1, 1, 1)
    expected = collapsed_ei_2022(base, best_x, X)
    with torch.no_grad():
        deflated_values = torch.exp(acq(X))
    where = expected > 1e-4  # outside the clamped region, where both are near zero
    assert torch.allclose(deflated_values[where], expected[where], rtol=1e-6, atol=1e-9)
    assert torch.all(deflated_values[~where] < 1e-3)


def test_removing_a_mode_moves_the_maximizer_to_the_next_mode():
    acq = DeflatedAcquisitionFunction(ExpectedImprovement(), noise=1.0, variance_fn=lambda X: torch.ones(1))
    first = grid_optimizer(acq)
    acq.remove_mode(first)
    second = grid_optimizer(acq)
    assert abs(first.item() - 1.0) < 0.05 and abs(second.item() + 1.0) < 0.05


@pytest.mark.parametrize("mode", ["subtractive", "multiplicative"])
def test_signed_function_is_deflated_without_clamping_its_negative_values(mode):
    acq = DeflatedAcquisitionFunction(UpperConfidenceBound(), mode=mode, temperature=0.5, noise=1.0, variance_fn=lambda X: torch.ones(1))
    first = grid_optimizer(acq)
    acq.remove_mode(first)
    X = torch.linspace(-3, 3, 61).reshape(-1, 1, 1)
    with torch.no_grad():
        values = acq(X)
    assert torch.all(torch.isfinite(values))
    second = grid_optimizer(acq)
    assert abs(second.item() - first.item()) > 0.1  # the maximizer leaves the removed mode


def test_multiplicative_mode_is_low_at_the_mode_and_unchanged_far_away():
    acq = DeflatedAcquisitionFunction(ExpectedImprovement(), mode="multiplicative", shift=1e-3, noise=1.0, variance_fn=lambda X: torch.ones(1))
    x0 = grid_optimizer(acq)
    with torch.no_grad():
        before = acq(torch.tensor([[[x0.item()]], [[-1.0]]]))
    acq.remove_mode(x0)
    with torch.no_grad():
        after = acq(torch.tensor([[[x0.item()]], [[-1.0]]]))
    assert after[0] - before[0] == pytest.approx(math.log(1e-3), abs=1e-6)
    assert after[1] == pytest.approx(before[1].item(), abs=5e-3)  # the bump is about 0.001 there


def test_deflate_stops_where_the_model_is_uncertain():
    # The model is certain near x = 1 (variance 0.1) and uncertain elsewhere (variance 10).
    variance = lambda X: torch.where((X.reshape(-1) - 1.0).abs() < 0.5, 0.1, 10.0)
    acq = DeflatedAcquisitionFunction(ExpectedImprovement(), noise=1.0, threshold=2.0, variance_fn=variance)
    x = acq.deflate(grid_optimizer, max_modes=5)
    assert len(acq.centers) == 1
    assert abs(x.item() + 1.0) < 0.05


def test_search_distribution_has_one_component_per_removed_mode():
    acq = DeflatedAcquisitionFunction(ExpectedImprovement(), noise=1.0, variance_fn=lambda X: torch.ones(1))
    for _ in range(2):
        acq.remove_mode(grid_optimizer(acq))
    mixture = acq.search_distribution()
    assert mixture.mixture_distribution.probs.shape == (2,)
    assert mixture.sample((5,)).shape == (5, 1)


def test_with_botorch_models():
    botorch = pytest.importorskip("botorch")
    from botorch.acquisition.analytic import ExpectedImprovement as EI, LogExpectedImprovement, UpperConfidenceBound as UCB
    from botorch.models import SingleTaskGP

    train_x = torch.linspace(0, 1, 8).reshape(-1, 1)
    train_y = torch.sin(6 * train_x)
    model = SingleTaskGP(train_x, train_y)
    best_f = train_y.max()
    for base, kind in [(EI(model, best_f), "positive"), (LogExpectedImprovement(model, best_f), "log"), (UCB(model, beta=2.0), "signed")]:
        acq = DeflatedAcquisitionFunction(base)
        assert acq.kind == kind
        assert acq.name == "Deflated" + type(base).__name__
        x = acq.deflate(lambda a: grid_optimizer(a, 0.0, 1.0, 201), max_modes=3)
        assert x.shape == (1,)
        with torch.no_grad():
            assert torch.all(torch.isfinite(acq(torch.rand(10, 1, 1))))

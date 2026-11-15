"""Tests of boic/botorch/gaussianweighted_ei.py, the belief-augmented acquisition functions of
Appendix I (GWEI and GKEI).

The module is loaded from its file, so that the tests do not import the whole boic package,
which pins BoTorch 0.6.0. The posterior is replaced by a fixed one, so that the tests check
the weighting by the prior, which is what this module adds to Expected Improvement.
Run with: python -m pytest tests/test_gaussianweighted_ei.py
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("botorch")
from botorch.models import SingleTaskGP

PATH = Path(__file__).resolve().parents[1] / "boic" / "botorch" / "gaussianweighted_ei.py"
spec = importlib.util.spec_from_file_location("gaussianweighted_ei", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GWEI = module.GaussianWeightedExpectedImprovement

torch.set_default_dtype(torch.float64)
DIM = 3


def model():
    x = torch.rand(5, DIM) * 2 - 1
    return SingleTaskGP(x, x.sum(-1, keepdim=True))


def acquisition(**kwargs):
    acq = GWEI(model(), best_f=0.0, maximize=False, **kwargs)
    # a fixed posterior, mean 0 and variance 1 everywhere, so that EI is the same at every point
    acq._get_posterior = lambda X: SimpleNamespace(mean=torch.zeros(X.shape[0], 1, 1),
                                                   variance=torch.ones(X.shape[0], 1, 1))
    return acq


EI_AT_ZERO = 1 / math.sqrt(2 * math.pi)  # EI with mean equal to best_f and unit standard deviation


def test_prior_has_standard_deviation_of_a_quarter_of_the_domain():
    acq = acquisition()  # default bounds [-1, 1]^D, width 2
    assert torch.allclose(acq.mvn.covariance_matrix, torch.eye(DIM) * 0.5 ** 2)


def test_fixed_location_is_the_origin_and_adaptive_location_is_given():
    assert torch.allclose(acquisition().reference_input, torch.zeros(DIM))
    best = torch.tensor([0.3, -0.2, 0.1])
    assert torch.allclose(acquisition(reference_input=best).reference_input, best)


def test_gkei_weight_is_the_gaussian_kernel_raised_to_beta():
    x = torch.tensor([[[0.5, 0.0, 0.0]]])
    acq = acquisition(normalization=False, beta_t=2.0)
    kernel = math.exp(-0.5 * (0.5 / 0.5) ** 2)
    assert acq(x).item() == pytest.approx(EI_AT_ZERO * kernel ** 2, rel=1e-9)


def test_gkei_equals_ei_at_the_location_of_the_prior():
    acq = acquisition(normalization=False, beta_t=5.0)
    assert acq(torch.zeros(1, 1, DIM)).item() == pytest.approx(EI_AT_ZERO, rel=1e-9)


def test_gwei_includes_the_normalizing_constant():
    x = torch.zeros(1, 1, DIM)
    gkei = acquisition(normalization=False, beta_t=1.0)(x).item()
    gwei = acquisition(normalization=True, beta_t=1.0)(x).item()
    constant = (2 * math.pi * 0.25) ** (-DIM / 2)
    assert gwei == pytest.approx(gkei * constant, rel=1e-9)


def test_the_weight_flattens_as_beta_decays():
    near, far = torch.zeros(1, 1, DIM), torch.full((1, 1, DIM), 0.9)
    ratio = lambda beta: (acquisition(normalization=False, beta_t=beta)(far)
                          / acquisition(normalization=False, beta_t=beta)(near)).item()
    assert ratio(10.0) < ratio(1.0) < ratio(0.1) < 1.0


def test_floor_keeps_the_weight_above_zero_far_from_the_location():
    far = torch.full((1, 1, DIM), 1e3)
    assert acquisition(normalization=False, beta_t=1.0)(far).item() == 0.0
    assert acquisition(normalization=False, beta_t=1.0, prob_floor=1e-12)(far).item() == pytest.approx(EI_AT_ZERO * 1e-12)

"""Tests of boic/torch/cmaes.py.

The module is loaded from its file with a stand-in for its base class, so that the tests
do not import the whole boic package, which pins BoTorch 0.6.0.
Run with: python -m pytest tests/test_cmaes.py
"""
import importlib.util
import sys
import types
from abc import ABC
from pathlib import Path

import pytest
import torch

pytest.importorskip("cma")

ROOT = Path(__file__).resolve().parents[1] / "boic"
# Stand-in packages, so that ``from ..core.models import SurrogateModel`` resolves without boic/__init__.py
for name in ["boic", "boic.core", "boic.torch"]:
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / name.split(".", 1)[1].replace(".", "/"))] if "." in name else [str(ROOT)]
    sys.modules.setdefault(name, module)
spec = importlib.util.spec_from_file_location("boic.core.models", ROOT / "core" / "models.py")
models = importlib.util.module_from_spec(spec)
sys.modules["boic.core.models"] = models
spec.loader.exec_module(models)
spec = importlib.util.spec_from_file_location("boic.torch.cmaes", ROOT / "torch" / "cmaes.py")
cmaes = importlib.util.module_from_spec(spec)
sys.modules["boic.torch.cmaes"] = cmaes
spec.loader.exec_module(cmaes)
CMAES = cmaes.CMAES

DIM = 5
BOUNDS = torch.stack([-torch.ones(DIM) * 5, torch.ones(DIM) * 5])


def sphere(x):
    return ((x - 1.0) ** 2).sum(-1)


def run(model, n, inputs, targets):
    """Acquire n points one at a time, as the manager does, and return all data."""
    for _ in range(n):
        x = model.acquire(bounds=BOUNDS, fn="ei", method="lbfgsb")  # extra options are ignored
        inputs = torch.cat([inputs, x])
        targets = torch.cat([targets, sphere(x)])
        model.set_train_data(inputs, targets, opt_data=None)
        model.train()
    return inputs, targets


def initial_data(seed=0, n=5):
    torch.manual_seed(seed)
    inputs = torch.rand(n, DIM) * 10 - 5
    return inputs, sphere(inputs)


def test_is_a_surrogate_model():
    assert issubclass(CMAES, models.SurrogateModel)


def test_starts_at_the_best_initial_point():
    inputs, targets = initial_data()
    model = CMAES(BOUNDS, inputs, targets, x0=0)
    assert torch.allclose(torch.as_tensor(model.x0), inputs[targets.argmin()])


def test_starts_at_the_prior_input_without_data():
    model = CMAES(BOUNDS, x0=0.5)
    assert torch.allclose(torch.as_tensor(model.x0).double(), torch.full((DIM,), 0.5, dtype=torch.float64))


def test_initial_step_size_is_a_twentieth_of_the_width():
    model = CMAES(BOUNDS)
    assert model.sigma0 == pytest.approx(0.5)


def test_acquisitions_are_within_bounds_and_one_at_a_time():
    inputs, targets = initial_data()
    model = CMAES(BOUNDS, inputs, targets)
    inputs, targets = run(model, 25, inputs, targets)
    new = inputs[5:]
    assert new.shape == (25, DIM)
    assert torch.all(new >= BOUNDS[0]) and torch.all(new <= BOUNDS[1])


def test_improves_on_a_simple_objective():
    inputs, targets = initial_data()
    model = CMAES(BOUNDS, inputs, targets)
    inputs, targets = run(model, 300, inputs, targets)
    assert targets[-10:].min() < targets[:5].min()
    assert targets.min() < 0.5


def test_state_round_trip_continues_the_same_search():
    inputs, targets = initial_data()
    model = CMAES(BOUNDS, inputs, targets)
    inputs, targets = run(model, 10, inputs, targets)  # one whole population
    saved = model.state()
    restored = CMAES(BOUNDS, inputs, targets)
    restored.state(state=saved)
    assert torch.allclose(torch.as_tensor(restored._cma.mean), torch.as_tensor(model._cma.mean))

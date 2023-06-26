import botorch

from botorch import acquisition, exceptions, models, optim, posteriors, settings, test_functions, utils
from botorch.models.gpytorch import GPyTorchModel as BoTorchModel
from botorch.cross_validation import batch_cross_validation
from botorch.fit import fit_gpytorch_model
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from botorch.utils import manual_seed, draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_scipy, fit_gpytorch_torch
from botorch.acquisition.analytic import ExpectedImprovement

from botorch.version import version as __version__

# patch gen_candidates_scipy
from .generate import gen_candidates_scipy
botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy

# patch _scipy_objective_and_grad
from .scipy_objective import _scipy_objective_and_grad
botorch.optim.utils._scipy_objective_and_grad = _scipy_objective_and_grad

# patch optimize_acqf
from .optimize import optimize_acqf, gen_batch_initial_conditions
botorch.optim.optimize.optimize_acqf = optimize_acqf
botorch.optim.optimize_acqf = optimize_acqf
botorch.optim.optimize.gen_batch_initial_conditions = gen_batch_initial_conditions
botorch.optim.gen_batch_initial_conditions = gen_batch_initial_conditions

__all__ = [
    "__version__",
    "BoTorchModel",
    "acquisition",
    "batch_cross_validation",
    "draw_sobol_samples",
    "exceptions",
    "ExpectedImprovement",
    "_scipy_objective_and_grad",
    "fit_gpytorch_model",
    "fit_gpytorch_scipy",
    "fit_gpytorch_torch",
    "gen_batch_initial_conditions",
    "gen_candidates_scipy",
    "gen_candidates_torch",
    "get_best_candidates",
    "manual_seed",
    "models",
    "optim",
    "optimize_acqf",
    "posteriors",
    "settings",
    "test_functions",
    "utils"
]
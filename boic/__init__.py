__name__ = 'boic'

from .core import Caster, FieldDict, AttributeFieldDict, ParsingDict
from .core import IO, RegisterIO, PendingIO, Job
from .core import Manager
from .core import Data
from .core import SurrogateModel
from .core import RunSettings, Settings
from .core import TestFn, TestDummy, TEST_FN_CHOICES
from .core import Aliaser, Method, Methods, Performance, MethodPerformance
from .core import AggregatedFrame, Results

from .botorch import gpytorch, BoTorchModel
from .botorch import acquisition, ExpectedImprovement
from .botorch import _scipy_objective_and_grad
from .botorch import fit_gpytorch_model, fit_gpytorch_scipy, fit_gpytorch_torch
from .botorch import gen_batch_initial_conditions, gen_candidates_scipy, gen_candidates_torch
from .botorch import optimize_acqf
from botorch.posteriors import GPyTorchPosterior

from .torch import TorchSettings, TorchMethod
from .torch import GPyTorchModel, ExactGPModel, GPTorchMethod, GPTorchSettings
from .torch import BoTorchSettings, BoTorchMethod, BoTorchPerformance
from .torch import TorchTestFn, TORCH_TEST_FN_CHOICES

from .utils import categorical_cmap, get_colors, get_colors_group
from .utils import get_unique_level_values, plot_df, plot_heatmap


__all__ = [# core
           'Caster',
           'FieldDict',
           'AttributeFieldDict',
           'ParsingDict',
           'IO',
           'RegisterIO',
           'PendingIO',
           'Job',
           'Manager',
           'Data',
           'SurrogateModel',
           'RunSettings',
           'Settings',
           'TestFn',
           'TEST_FN_CHOICES',
           'TestDummy',
           'Aliaser',
           'Method',
           'Methods',
           'Performance',
           'MethodPerformance',
           'AggregatedFrame',
           'Results',
           # botorch
           'gpytorch',
           'BoTorchModel',
           'acquisition',
           'ExpectedImprovement',
           '_scipy_objective_and_grad',
           'fit_gpytorch_model',
           'fit_gpytorch_scipy',
           'fit_gpytorch_torch',
           'gen_batch_initial_conditions',
           'gen_candidates_scipy',
           'gen_candidates_torch',
           'optimize_acqf',
           'GPyTorchPosterior',
           # torch
           'TorchSettings',
           'TorchMethod',
           'GPyTorchModel',
           'ExactGPModel',
           'GPTorchMethod',
           'GPTorchSettings',
           'BoTorchSettings',
           'BoTorchMethod',
           'BoTorchPerformance',
           'TorchTestFn',
           'TORCH_TEST_FN_CHOICES',
           # utils
           "categorical_cmap",
           "get_colors",
           "get_colors_group",
           "get_unique_level_values",
           "plot_df",
           "plot_heatmap"
           ]
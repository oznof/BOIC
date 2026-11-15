"""CMA-ES as a model of the experiment framework, the evolutionary baseline of the paper.

CMA-ES (covariance matrix adaptation evolution strategy, Hansen, 2016) is run through
pycma (``cma``). It proposes a population of points at a time: ``acquire`` returns them
one by one, and once a whole population has been evaluated, the results are passed to
CMA-ES to update its search distribution. It has no hyperparameters to train.

Select it with ``-model_mode Kcmaes``. The initial mean is the prior input, for example
the center of the search space with ``-prior_mode Xcenter``, or the best point already
evaluated, and the initial step size is one twentieth of the search space's width.
"""
import pickle

import torch

from ..core.models import SurrogateModel


class CMAES(SurrogateModel):
    def __init__(self, bounds, train_inputs=None, train_targets=None, x0=0, sigma0=None,
                 popsize=10, options=None):
        from cma import CMAEvolutionStrategy  # optional dependency, imported only when used
        self._bounds = bounds
        self._x0 = x0
        self._sigma0 = sigma0
        self.set_train_data(train_inputs, train_targets)
        options = options or {}
        options.setdefault('popsize', popsize)
        options.setdefault('verb_disp', 0)
        options['bounds'] = self.bounds
        self._options = options
        self._cma = CMAEvolutionStrategy(self.x0, self.sigma0, self.options)
        self._batch_acquisition = []
        self._made_acquisitions = False

    @property
    def bounds(self):
        return torch.as_tensor(self._bounds).numpy().tolist()

    @property
    def dim(self):
        return len(self.bounds[0])

    @property
    def x0(self):
        if self.train_inputs is not None and self.train_targets is not None:
            x0 = self.train_inputs[self.train_targets.argmin()]
        else:
            x0 = torch.ones(self.dim) * torch.as_tensor(self._x0).reshape(-1)
        return x0.numpy()

    @property
    def sigma0(self):
        if self._sigma0 is None:
            lb, ub = torch.as_tensor(self.bounds)
            self._sigma0 = ((ub - lb) / 20).numpy().mean()
        return self._sigma0

    @property
    def options(self):
        return self._options

    @property
    def nbatch(self):
        return self.options['popsize']

    @property
    def batch_acquisition(self):
        return self._batch_acquisition

    @property
    def made_acquisitions(self):
        return self._made_acquisitions

    def set_train_data(self, inputs=None, targets=None, **kwargs):
        self.train_inputs = torch.atleast_2d(torch.as_tensor(inputs)) if inputs is not None else inputs
        self.train_targets = torch.as_tensor(targets).reshape(-1) if targets is not None else targets

    def train(self, *args, info_to_numpy=True, **kwargs):
        self.train_info = None

    def state(self, state=None, copy=False, to_numpy=True):
        params = {} if state is None else state['params']
        if params:
            self._cma = pickle.loads(params)
        return {'params': self._cma.pickle_dumps()}

    def acquire(self, bounds, **kwargs):
        assert torch.equal(torch.as_tensor(bounds), torch.as_tensor(self.bounds))
        if len(self._batch_acquisition) == 0:
            if self.made_acquisitions:
                n = self.nbatch
                self._cma.tell(self.train_inputs.numpy()[-n:], self.train_targets.numpy()[-n:])
            # prepare a new batch, given by the population
            self._batch_acquisition = self._cma.ask()
        acquisition = torch.atleast_2d(torch.as_tensor(self.batch_acquisition.pop()))
        self._made_acquisitions = True
        return acquisition

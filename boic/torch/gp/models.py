import copy
import math

import numpy as np
from matplotlib import pyplot as plt

import torch

from boic.botorch import fit_gpytorch_scipy, fit_gpytorch_torch, BoTorchModel
from boic.botorch import gpytorch, optimize_acqf, ExpectedImprovement
from boic.botorch.gpytorch import settings
from boic.botorch.gpytorch.kernels import Kernel
from boic.botorch.gpytorch.priors import Prior, SmoothedBoxPrior, UniformPrior, HalfHorseshoePrior, KumaraswamyPrior
from boic.botorch.gpytorch.constraints import Interval, Positive
from boic.botorch.gpytorch import MultivariateNormal
from boic.botorch.gpytorch import means
from boic.botorch.gpytorch.mll import ExactMarginalLogLikelihood
from boic.botorch.gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, InformativeKernel, CylindricalKernel
from boic.botorch.gpytorch.kernels.cylindrical import _original_prediction_strategy_cylindrical
from boic.botorch.gpytorch.constraints.constraints import softplus, inv_softplus
from boic.botorch.scipy_objective import _scipy_objective_and_grad

from boic.core import SurrogateModel


NV_LB, NV_UB = math.exp(-12), math.exp(16)
SV_LB, SV_UB = math.exp(-12), math.exp(20)

kernel_cls = {'matern': MaternKernel, 'rbf': RBFKernel}
interval = lambda *args, **kwargs: Interval(*args, transform=softplus, inv_transform=inv_softplus, **kwargs)


class GPyTorchModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, mean_module, covar_module):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactGPModel(SurrogateModel, BoTorchModel, gpytorch.models.ExactGP):
    r"""Exact Gaussian process model as implemented in Appendix B of https://arxiv.org/abs/2208.02704

    :param anchor: The location of the anchor :math:`\mathbf{x}_0` (Optional).
          Default: :math:`\mathbf{0}`
    :type  anchor: Number, torch.Tensor, np.ndarray
    :param train_inputs: Training inputs :math:`\mathbf X` (size n x d).
          Default: ``None``
    :type  train_inputs: torch.Tensor
    :param train_targets: Training targets :math:`\mathbf y` (size n x d).
          Default: ``None``
    :type  train_targets: torch.Tensor
    :param mean_options: Mean function options. For more details, see setter.
    :param kernel_options: Kernel function options. For more details, see setter.
    :param likelihood_options: Likelihood options. For more details, see setter.
    :param tr_options: Trust region options. For more details, see setter.
    """
    def __init__(self, anchor=None,
                 train_inputs=None, train_targets=None,
                 mean_options=None, kernel_options=None, likelihood_options=None,
                 tr_options=None, **kwargs):

        if isinstance(train_inputs, list):
            train_inputs = train_inputs[0]
        self._dim = train_inputs.size(-1) if isinstance(train_inputs, torch.Tensor) else kwargs['dim']
        self._train_inputs = train_inputs
        self._train_targets = train_targets
        self._kwargs = kwargs

        self.anchor = anchor
        self.mean_options = mean_options
        self.kernel_options = kernel_options
        self.likelihood_options = likelihood_options
        self.tr_options = tr_options

        likelihood = gpytorch.likelihoods.GaussianLikelihood(**self.likelihood_options)
        if self.likelihood_options['fixed_noise'] is not None:
            likelihood.noise_covar.noise = self.likelihood_options['fixed_noise']
            likelihood.noise_covar.raw_noise.requires_grad = False

        gpytorch.models.ExactGP.__init__(self, train_inputs, train_targets, likelihood)

        if self.mean_options['mode'] == 'const':
            self.mean_module = means.ConstantMean(prior=self.mean_options['constant_prior'])
        elif self.mean_options['mode'].startswith('quad'):
            self.mean_module = means.QuadMean(input_size=self.dim, anchor=self.anchor, **self.mean_options)
        else:
            raise NotImplementedError

        kernel_options = copy.deepcopy(self.kernel_options)
        kernel = kernel_options.pop('kernel')
        base_kernel = kernel_options.pop('base_kernel')
        kernel_mode = kernel_options.pop('mode')

        if kernel == 'sk' and kernel_mode == 'base':
            self.covar_module = ScaleKernel(kernel_cls[base_kernel](**kernel_options), **kernel_options)
        elif kernel == 'ck' and kernel_mode == 'base':
            radial_base_kernel = ScaleKernel(kernel_cls[base_kernel](**kernel_options), **kernel_options)
            self.covar_module = CylindricalKernel(radial_base_kernel=radial_base_kernel, **kernel_options)
        elif kernel == 'ik':
            if kernel_mode not in ['fixed', 'greedy']:
                raise ValueError(f'For InformativeKernel only fixed and greedy are implemented.')
            self.covar_module = InformativeKernel(anchor=self.anchor, base_kernel=kernel_cls[base_kernel],
                                                  **kernel_options)
            # self.lengthscale_variance_init = self.covar_module.lengthscale_variance.detach().clone() \
            #     if self.covar_module.has_nonstationary_variance else None
            self.ratio_variance_init = self.covar_module.ratio_variance.detach().clone() \
                if self.covar_module.has_nonstationary_variance else None
            # self.lengthscale_lengthscale_init = self.covar_module.lengthscale_lengthscale.detach().clone() \
            #     if self.covar_module.has_nonstationary_lengthscale else None
            self.ratio_lengthscale_init = self.covar_module.ratio_lengthscale.detach().clone()\
                if self.covar_module.has_nonstationary_lengthscale else None
        else:
            raise ValueError(f'{kernel} and {kernel_mode} are not a valid kernel and kernel_mode for {self.__class__}')
        self._num_outputs = 1
        kernel_init = {}
        if 'lengthscale' in kwargs:
            kernel_init['lengthscale'] = kwargs['lengthscale']
        if 'outputscale' in kwargs:
            kernel_init['outputscale'] = kwargs['outputscale']
        if kernel_init:
            self.kernel_params(**kernel_init)
        self.curr_best_input = None
        self.curr_best_target = None
        self.tr_scount = 0
        self.tr_fcount = 0
        self.acquisition = {}  # store here any info that may be useful, e.g.,
                               # depending on the acquisition function one may set different behavior
                               # when setting data and initializing any parameters of the model
        if train_inputs is not None and train_targets is not None:
            self.set_train_data(self.train_inputs, self.train_targets)

    @property
    def device(self):
        if isinstance(self.train_inputs, torch.Tensor):
            return self.train_inputs.device

    @property
    def dim(self):
        return self._dim

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        if value is None:
            value = 0
        if isinstance(value, int) or isinstance(value, float):
            value = float(value) * torch.ones(1, 1 if self.dim is None else self.dim)
        elif isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
            if len(value) == 1:
                value *= torch.ones(self.dim)
        self._anchor = value

    def anchor(self, value):
        if value is None:
            value = 0
        if isinstance(value, int) or isinstance(value, float):
            value = float(value) * torch.ones(*self.batch_shape, 1, self.dim)
        value = self.as_tensor(value)
        if len(value) == 1:
            value *= torch.ones(self.dim)
        self._anchor = value
    @property
    def settings(self):
        settings = {'anchor': self.anchor,
                    'mean_options': self.mean_options, 'kernel_options': self.kernel_options,
                    'likelihood_options': self.likelihood_options, 'tr_options': self.tr_options}
        settings.update(self._kwargs)
        return settings

    @property
    def mean_options(self):
        return self._mean_options

    @mean_options.setter
    def mean_options(self, value):
        value = {} if not isinstance(value, dict) else copy.deepcopy(value)
        value.setdefault('greedy_anchor', False)  # by default, the anchor in the mean function is fixed
        value.setdefault('mode', 'const')  # options: 'const', 'quad'
        value.setdefault('constant_prior', UniformPrior(1e-6, 1e6))  # by default, this will change to empirical prior
        value.setdefault('constant_constraint', None)                # see method set_train_data
        if value['mode'] == 'quad':
            value.setdefault('a_prior', HalfHorseshoePrior(scale=2, validate_args=False))
            value.setdefault('a_constraint', Positive())
        self._mean_options = value

    @property
    def ratio_variance_prior(self):
        return self._ratio_prior('ratio_variance_prior')

    @property
    def ratio_lengthscale_prior(self):
        return self._ratio_prior('ratio_lengthscale_prior')

    def _ratio_prior(self, prior_name):
        prior = self._kernel_options.get(prior_name)
        if not isinstance(prior, Prior):
            if prior == 'k':
                # weakly informative, peak @ 0.1
                prior = KumaraswamyPrior(1.467, 10, validate_args=False)
            elif prior == 'ki':
                # more informative/narrow, peak @ 0.1
                prior = KumaraswamyPrior(2.253, 100, validate_args=False)
            elif prior is None or prior == 'kii':
                # even more informative, peak @ 0.1
                prior = KumaraswamyPrior(3.164, 1000, validate_args=False)
            elif prior == 'u':
                prior = UniformPrior(1e-6, 1 + 1e-6, validate_args=False)
            else:
                raise NotImplementedError
        return prior

    @property
    def kernel_options(self):
        return self._kernel_options

    @kernel_options.setter
    def kernel_options(self, value):
        self._kernel_options = value if isinstance(value, dict) else {}
        LS_LB, LS_UB, LS_INIT = math.exp(-12), 2 * math.sqrt(self.dim), math.sqrt(self.dim)
        value = {} if not isinstance(value, dict) else copy.deepcopy(value)
        value.setdefault('ard', True)
        value.setdefault('ard_num_dims', self.dim if value['ard'] else 1)
        value.setdefault('dim', self.dim)
        value.setdefault('kernel', 'sk')  # options: 'sk', 'ck', 'ik'
        value.setdefault('base_kernel', 'matern')  # options: 'matern', 'rbf'
        value.setdefault('mode', 'base')  # options: 'base', 'fixed', 'greedy'
        value.setdefault('lengthscale_prior', UniformPrior(LS_LB, LS_UB, validate_args=False))
        value.setdefault('lengthscale_constraint', interval(LS_LB, LS_UB, initial_value=LS_INIT))
        value.setdefault('outputscale_prior', UniformPrior(SV_LB, SV_UB, validate_args=False))
        outputscale_init = 1. if self._train_targets is None else self._train_targets.std().item()
        value.setdefault('outputscale_constraint', interval(SV_LB, SV_UB,
                                                            initial_value=outputscale_init))
        if value['kernel'] == 'ik':
            R_LB, R_UB, R_INIT = 1e-6, 1 + 1e-6, 0.1
            value.setdefault('pool', True)
            value.setdefault('requires_grad', False)  # overrides all requires_grad flags if True
            value.setdefault('has_nonstationary_variance', True)
            value.setdefault('lengthscale_variance', 's')  # shared lengthscales
            value.setdefault('lengthscale_variance_requires_grad', False)
            value.setdefault('lengthscale_variance_constraint', interval(LS_LB, LS_UB, initial_value=LS_INIT))
            value.setdefault('lengthscale_variance_prior', UniformPrior(LS_LB, LS_UB, validate_args=False))
            value.setdefault('ratio_variance', R_INIT)
            value.setdefault('ratio_variance_requires_grad', True)
            value.setdefault('ratio_variance_constraint', interval(R_LB, R_UB, initial_value=R_INIT))
            value['ratio_variance_prior'] = self.ratio_variance_prior
            value.setdefault('has_nonstationary_lengthscale', True)
            value.setdefault('lengthscale_lengthscale', 's')  # shared lengthscales
            value.setdefault('lengthscale_lengthscale_requires_grad', False)
            value.setdefault('lengthscale_lengthscale_constraint', interval(LS_LB, LS_UB, initial_value=LS_INIT))
            value.setdefault('lengthscale_lengthscale_prior', UniformPrior(LS_LB, LS_UB, validate_args=False))
            value.setdefault('ratio_lengthscale', R_INIT)
            value.setdefault('ratio_lengthscale_requires_grad', True)
            value.setdefault('ratio_lengthscale_constraint', interval(R_LB, R_UB, initial_value=R_INIT))
            value['ratio_lengthscale_prior'] = self.ratio_lengthscale_prior
        self._kernel_options = value

    @property
    def likelihood_options(self):
        return self._likelihood_options

    @likelihood_options.setter
    def likelihood_options(self, value):
        value = {} if not isinstance(value, dict) else copy.deepcopy(value)
        value.setdefault('fixed_noise', None)
        value.setdefault('noise_prior', UniformPrior(NV_LB, NV_UB, validate_args=None))
        value.setdefault('noise_constraint', interval(1e-6, 1e6,
                                                      initial_value=value['fixed_noise'] or 1e-1))
        self._likelihood_options = value

    @property
    def tr_options(self):
        return self._tr_options

    @tr_options.setter
    def tr_options(self, value):
        value = {} if not isinstance(value, dict) else copy.deepcopy(value)
        value.setdefault('stol', 3)
        value.setdefault('ftol', 10)
        value.setdefault('length_min', 0.5 ** 7)
        value.setdefault('length_max', 1.6)
        value.setdefault('length_init', 0.8)
        value.setdefault('sfactor', 1e-3)
        value['length'] = value['length_init']
        self._tr_options = value

    def as_tensor(self, value):
        return torch.tensor(value, dtype=torch.get_default_dtype(), device=self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        if isinstance(self.covar_module, CylindricalKernel) and self.covar_module.original:
            if self.training:
                # NOTE: This is implemented in the imported ExactMarginalLogLikelihood class, do nothing here
                return
            if settings.prior_mode.off():
                return _original_prediction_strategy_cylindrical(self, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def set_train_data(self, inputs=None, targets=None, strict=False, opt_data=None):
        gpytorch.models.ExactGP.set_train_data(self, inputs, targets, strict)
        if targets is not None:
            constant_prior_name = 'mean_prior' if self.mean_options['mode'] == 'const' else 'constant_prior'
            if hasattr(self.mean_module, constant_prior_name):
                constant_prior = getattr(self.mean_module, constant_prior_name)
                if constant_prior.__class__ in [SmoothedBoxPrior, UniformPrior]:
                    constant_prior = constant_prior.__class__(targets.min(), targets.max(), validate_args=False)
                    self.mean_module.register_prior(constant_prior_name, constant_prior, 'constant')
                    self.mean_module.initialize(constant=targets.mean())
        if self.train_targets is not None:
            if self.kernel_options['mode'] == 'greedy' or self.mean_options['greedy_anchor']:
                train_inputs = self.train_inputs
                if isinstance(train_inputs, tuple):
                    train_inputs = train_inputs[0]
                best_input = train_inputs[int(self.train_targets.argmin())].detach().clone()
                if isinstance(self.covar_module, InformativeKernel) and self.kernel_options['mode'] == 'greedy':
                    if not torch.equal(self.covar_module.anchor, best_input):
                        self.covar_module.anchor = best_input
                        if self.covar_module.has_nonstationary_variance:
                            if self.covar_module.ratio_variance.requires_grad and self.ratio_variance_init is not None:
                                self.covar_module.ratio_variance = self.ratio_variance_init
                        if self.covar_module.has_nonstationary_lengthscale and not self.covar_module.pool:
                            if self.covar_module.ratio_lengthscale.requires_grad and self.ratio_lengthscale_init is not None:
                                self.covar_module.ratio_lengthscale = self.ratio_lengthscale_init
                if isinstance(self.mean_module, means.QuadMean) and self.mean_options['greedy_anchor']:
                    best_input = best_input.expand_as(self.mean_module.anchor)
                    # compare first because if different we want to reset quadratic weights
                    if not torch.equal(best_input, self.mean_module.anchor):
                        self.mean_module.anchor = best_input
                        self.mean_module.a = 1
        if isinstance(self.covar_module, InformativeKernel):
            self.covar_module.update_nonstationary_lengthscales()
        if opt_data is not None and opt_data.has_data:
            # opt_data is required for trust region optimization
            self.acquisition['data'] = opt_data
            if opt_data.n == 0:
                best_id = opt_data.argbest(opt_data.train_targets)
                self.curr_best_input = opt_data.train_inputs[best_id]
                self.curr_best_target = opt_data.train_targets[best_id]
            else:
                new_input = opt_data.train_inputs[-1]
                new_target = opt_data.train_targets[-1]
                #if self.curr_best_target is None: # todo: model is loading from checkpoint (not used for now w/ tr)
                #    return
                target_diff = opt_data.diff(self.curr_best_target, new_target)
                is_better = target_diff > 0
                is_success = (target_diff / np.abs(self.curr_best_target)) >= self.tr_options['sfactor']
                if is_success:
                    self.tr_scount += 1
                    self.tr_fcount = 0
                else:
                    self.tr_scount = 0
                    self.tr_fcount += 1
                if is_better:
                    self.curr_best_input = new_input
                    self.curr_best_target = new_target
                if self.tr_scount == self.tr_options['stol']:
                    self.tr_scount = 0
                    self.tr_options['length'] = min([2 * self.tr_options['length'], self.tr_options['length_max']])
                elif self.tr_fcount == self.tr_options['ftol']:
                    self.tr_fcount = 0
                    length = 0.5 * self.tr_options['length']
                    if length < self.tr_options['length_min']:
                        length = self.tr_options['length_init']
                    self.tr_options['length'] = length

    def train(self, *args, **kwargs):
        if (not args and not kwargs) or (args and isinstance(args[0], bool)) or\
           ('mode' in kwargs and isinstance(kwargs['mode'], bool)):
            return BoTorchModel.train(self, *args, **kwargs)
        else:
            return self._train(*args, **kwargs)

    def _train(self, method='L-BFGS-B', options=None, track_iterations=False,
               only_keep_last_iteration=True, **kwargs):
        method = method.lower().replace('-', '')
        if options is None:
            options = {}
        options.setdefault('maxiter', 1000)
        options.setdefault('disp', False)
        options.setdefault('lr', 0.01)
        self.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        if method == 'lbfgsb':
            scipy_objective = _scipy_objective_and_grad
            fit_fn = lambda *args, **kwargs: fit_gpytorch_scipy(*args, scipy_objective=scipy_objective, **kwargs)
            options.pop('lr')
        elif method == 'adam':
            fit_fn = fit_gpytorch_torch
        else:
            raise ValueError(f'{method} is an invalid train method.')
        _, info_dict = fit_fn(mll, options=options, track_iterations=track_iterations, approx_mll=False)
        if track_iterations and only_keep_last_iteration:
            try:
                info_dict['iterations'] = info_dict['iterations'][-1]
            except (KeyError, TypeError):
                pass
        return info_dict

    def predict(self, test_x, grad_enabled=False, make_plot=False, ax=None, with_noise=False):
        self.eval()
        with torch.set_grad_enabled(grad_enabled):
            ppd = lambda x: self(x)
            if with_noise:
                ppd = lambda x: self.likelihood(ppd(x))
            res = observed_pred = ppd(test_x)
            if make_plot:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(8, 5))
                lower, upper = observed_pred.confidence_region()
                ax.plot(self.train_inputs[0].numpy(), self.train_targets.numpy(), 'ko')
                ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
                ax.set_ylim([lower.min().floor().item(),
                             upper.max().ceil().item()])
                ax.grid()
                res = (res, ax)
            return res

    def sample_prior(self, test_x, n=1, grad_enabled=False,
                     with_noise=False, make_plot=False, get_prior=False):
        self.eval()
        with gpytorch.settings.prior_mode(True):
            ppd = self(test_x)
            if with_noise:
                ppd = self.likelihood(ppd)
            samples = ppd.sample(torch.Size([n, ]))
            res = samples
            if get_prior:
                res = [res, ppd]
            if make_plot:
                with torch.set_grad_enabled(grad_enabled):
                    f, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.plot(test_x.numpy(), samples.numpy().T)
                    ax.set_ylim([samples.min().floor().item(),
                                 samples.max().ceil().item()])
                    ax.grid()
                    if not isinstance(res, list):
                        res = list(res)
                    res.append(ax)
        return res

    def load_parameter(self, module, name, value=None, pdict=None):
        if pdict is None:
            pdict = {}
        if value is not None:
            if f'raw_{name}_constraint' in module._constraints:
                c = module._constraints[f'raw_{name}_constraint']
                if isinstance(c, Interval):
                    value = self.as_tensor(value)
                    value.clamp_(min=c.lower_bound, max=c.upper_bound)
            try:
                setattr(module, name, value)
            except TypeError:
                module.initialize(**{name:value})
        pdict[name] = getattr(module, name)
        return pdict

    def params(self, copy=False, to_numpy=False, **kwargs):
        params = {}
        params['kernel'] = self.kernel_params(**kwargs.get('kernel', {}))
        params['mean'] = self.mean_params(**kwargs.get('mean', {}))
        params['likelihood'] = self.likelihood_params(**kwargs.get('likelihood', {}))
        if copy or to_numpy:
            for k, v in params.items():
                for kk, vv in v.items():
                    if vv is not None:
                        data = torch.as_tensor(vv).cpu().detach().clone()
                        if to_numpy:
                            data = data.numpy()
                        params[k][kk] = data
        return params

    def kernel_params(self, outputscale=None, lengthscale=None,
                      lengthscale_variance=None, ratio_variance=None,
                      lengthscale_lengthscale=None, ratio_lengthscale=None,
                      anchor=None, angular_weights=None,
                      alpha=None, beta=None):
        module = self.covar_module
        params = {}
        if isinstance(module, InformativeKernel):
            if module.has_nonstationary_variance:
                self.load_parameter(module, 'lengthscale_variance', lengthscale_variance, params)
                self.load_parameter(module, 'ratio_variance', ratio_variance, params)
            if module.has_nonstationary_lengthscale:
                if not (self.kernel_options['pool'] and module.has_nonstationary_variance):
                    # don't set if parameters are pooled and ns variance exists
                    self.load_parameter(module, 'lengthscale_lengthscale', lengthscale_lengthscale, params)
                    self.load_parameter(module, 'ratio_lengthscale', ratio_lengthscale, params)
            if module.has_nonstationary_variance or module.has_nonstationary_lengthscale:
                self.load_parameter(module, 'anchor', anchor, params)
            module = module.base_kernel
        elif isinstance(module, CylindricalKernel):
            self.load_parameter(module, 'angular_weights', angular_weights, params)
            self.load_parameter(module, 'alpha', alpha, params)
            self.load_parameter(module, 'beta', beta, params)
            module = module.base_kernel
        if isinstance(module, ScaleKernel):
            self.load_parameter(module, 'outputscale', outputscale, params)
            module = module.base_kernel
        if isinstance(module, Kernel) and module.has_lengthscale:
            self.load_parameter(module, 'lengthscale', lengthscale, params)
        return params

    def mean_params(self, constant=None, a=None, anchor=None, **kwargs):
        module = self.mean_module
        params = {}
        if isinstance(constant, np.ndarray):
            constant = float(constant)
        self.load_parameter(module, 'constant', constant, params)
        if self.mean_options['mode'] == 'quad':
            self.load_parameter(module, 'a', a, params)
            self.load_parameter(module, 'anchor', anchor, params)
        return params

    def likelihood_params(self, noise=None, **kwargs):
        params = {}
        if noise:
            self.likelihood.noise_covar.initialize(noise=float(noise))
        params['noise'] = self.likelihood.noise_covar.noise
        return params

    def state(self, state=None, copy=False, to_numpy=True):
        param_kwargs = {} if state is None else state['params']
        return {'params': self.params(copy=copy, to_numpy=to_numpy, **param_kwargs)}

    def acquire(self, bounds, fn='trgreedyei', method='L-BFGS-B', num_restarts=20, raw_samples=20000, spray_samples=10,
                perturb_train=False, options=None):
        # Note: To enable greedy trust region, use 'trgreedy' + fn
        fn = fn.lower()
        self.acquisition['name'] = fn
        self.acquisition['other_settings'] = {'bounds': bounds, 'method': method,
                                              'num_restarts': num_restarts, 'raw_samples': raw_samples,
                                              'spray_samples': spray_samples, 'perturb_train': perturb_train,
                                              'options': options}

        best_id = torch.argmin(self.train_targets).item()
        train_inputs = self.train_inputs if isinstance(self.train_inputs, torch.Tensor) else self.train_inputs[0]
        best_input = train_inputs[best_id]
        best_target = self.train_targets[best_id]
        num_train = len(self.train_targets)
        _fn = fn.replace('trgreedy', '')
        if _fn == 'ei':
            acq_fn = ExpectedImprovement
            acqf = acq_fn(self, best_f=best_target, maximize=False)
            # NOTE: setting acqf as an attribute leads to recursionerror in train
            self.acquisition['fn'] = acqf
        else:
            raise ValueError(f"For this botorch model, {fn} is an invalid acquisition function.")
        if fn.startswith('trgreedy'): # enable greedy trust region
            bounds = self.update_tr_bounds()
        bounds = torch.tensor(bounds, dtype=torch.get_default_dtype(), device=self.device)
        method = method.lower().replace('-', '')
        if method == 'lbfgsb':
            additional_samples = torch.tensor([])
            if perturb_train:
                additional_samples = torch.cat((additional_samples, self.gen_spray_samples(train_inputs, bounds, 10)))
            if spray_samples > 0:
                additional_samples = torch.cat((additional_samples,
                                                self.gen_spray_samples(best_input, bounds, spray_samples)))
            additional_samples = None if len(additional_samples) == 0 else additional_samples.unsqueeze(1)
            new_input, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=num_restarts,
                                         raw_samples=raw_samples, additional_samples=additional_samples,
                                         options=options, precision=None)
        else:
            raise NotImplementedError
        return new_input

    @classmethod
    def gen_spray_samples(cls, inputs, bounds, n):
        inputs = torch.atleast_2d(inputs)
        lb, ub = bounds
        ni, dim = inputs.shape
        samples = inputs.repeat(n, 1) + inputs.new(n * ni, dim).normal_() * 0.001 * (ub - lb)
        samples = cls.enforce_bounds(samples, bounds)
        return samples

    @classmethod
    def enforce_bounds(cls, samples, bounds):
        samples = torch.atleast_2d(samples)
        n = samples.shape[0]
        lb, ub = bounds
        outl, outu = torch.tensor([True, True])
        while outl.any() or outu.any():
            outl = samples < lb
            samples[outl] = 2 * lb.view(1, -1).repeat(n, 1)[outl] - samples[outl]
            outu = samples > ub
            samples[outu] = 2 * ub.view(1, -1).repeat(n, 1)[outu] - samples[outu]
        return samples

    def update_tr_bounds(self):
        bounds = self.acquisition['other_settings']['bounds']
        lb, ub = torch.as_tensor(bounds).numpy()
        input_center = torch.as_tensor(self.curr_best_input).cpu().detach().numpy().ravel()
        length = self.tr_options['length']
        # make sure the lengthscale vector has dim entries to handle the case where ard is disabled
        weights = torch.as_tensor(self.kernel_params()['lengthscale']).cpu().detach().numpy().ravel() * np.ones(len(lb))
        weights = weights / weights.mean()
        weights = weights / np.prod(np.power(weights, 1 / len(weights))) # weights.prod() = 1
        lb = np.clip(input_center - weights * length / 2, a_min=lb, a_max=None)
        ub = np.clip(input_center + weights * length / 2, a_min=None, a_max=ub)
        self.acquisition['tr_bounds'] = tr_bounds = np.vstack([lb, ub])
        #print(f"UPDATE TR: length={length}, weights={weights}, lb={lb}, ub={ub}")
        return tr_bounds

    def acquisition_function(self, fn='ei', bounds=None):
        if fn.lower() == 'ei':
            best_id = torch.argmin(self.train_targets).item()
            best_target = self.train_targets[best_id]
            acq_fn = ExpectedImprovement(self, best_f=best_target, maximize=False)
        else:
            raise NotImplementedError
        return acq_fn

    def to_gpytorch_model(self):
        train_inputs = self.train_inputs[0] if isinstance(self.train_inputs, list) else self.train_inputs
        return GPyTorchModel(train_inputs=train_inputs, train_targets=self.train_targets, likelihood=self.likelihood,
                             mean_module=self.mean_module, covar_module=self.covar_module)

import os
import math

import numpy as np

import torch
from gpytorch.constraints.constraints import softplus, inv_softplus

from boic.torch.gp.settings import GPTorchSettings
from boic.botorch import gpytorch
from boic.torch.gp.models import ExactGPModel
from boic.botorch.gpytorch import constraints, priors
from boic.botorch.gpytorch.priors import HalfHorseshoePrior, KumaraswamyPrior, LogNormalPrior,\
                                         LogSpikeUniformSlabPrior, SpikeUniformSlabPrior
# todo: positive is causing error pickling, import it from patch file for now
from boic.botorch.gpytorch.constraints import Positive
from boic.botorch import botorch
from pulse.caster import Caster


Interval = constraints.Interval
UniformPrior = priors.UniformPrior

interval = lambda *args, **kwargs: Interval(*args, transform=softplus, inv_transform=inv_softplus, **kwargs)
interval_exp = lambda *args, **kwargs: Interval(*args, transform=torch.exp, inv_transform=torch.log, **kwargs)

class BoTorchSettings(GPTorchSettings):
    DEFAULT_PATH = os.path.join(GPTorchSettings.DEFAULT_PATH, 'botorch')
    DEFAULT_TRAIN_METHOD = 'lbfgsb'
    TRAIN_METHOD_CHOICES = ['lbfgsb']
    DEFAULT_TRAIN_METHOD_MODE = 'botorch'
    TRAIN_METHOD_MODE_CHOICES = ['botorch']
    ACQ_FN_CHOICES = ['ei']
    ACQ_FN_CHOICES += ['trgreedy' + afn for afn in ACQ_FN_CHOICES]
    DEFAULT_ACQ_FN = 'ei'
    DEFAULT_ACQ_METHOD = 'lbfgsb'
    ACQ_METHOD_CHOICES = ['lbfgsb']
    DEFAULT_ACQ_METHOD_MODE = 'botorch'
    ACQ_METHOD_MODE_CHOICES = ['botorch']
    DEFAULT_MODEL_MEAN_MODE = 'const'
    MODEL_MEAN_MODE_CHOICES = ['const', 'quad', 'quadu', 'quadhs', 'quadln',
                                                'quadfu', 'quadfhs', 'quadfln']
    DEFAULT_MODEL_KERNEL = 'sk'
    MODEL_KERNEL_CHOICES = ['sk', 'ck', 'ik']
    DEFAULT_MODEL_KERNEL_MODE = 'base'
    MODEL_KERNEL_MODE_CHOICES = ['base', 'fixed', 'greedy']

    def packages_initialize(self):
        super().packages_initialize()
        # check https://docs.gpytorch.ai/en/stable/settings.html for more settings
        # standard inference using cholesky decomposition
        gpytorch.settings._fast_covar_root_decomposition._state = False
        gpytorch.settings._fast_log_prob._state = False
        gpytorch.settings._fast_solves._state = False
        gpytorch.settings.fast_pred_var._state = False
        gpytorch.settings.fast_pred_samples._state = False
        gpytorch.settings.lazily_evaluate_kernels._state = True
        gpytorch.settings.num_trace_samples._global_value = 0
        botorch.settings.validate_input_scaling(False)

    @property
    def train_mode(self):
        return self['train']['mode']

    @train_mode.setter
    def train_mode(self, value):
        if self['train']['method_mode'] == 'botorch':
            self['train']['options'] = {'method': self['train']['method'], 'options': {'maxiter': 1000},
                                        'track_iterations': False}
        else:
            raise NotImplementedError

    @property
    def acq_mode(self):
        return self['acq']['mode']

    @acq_mode.setter
    def acq_mode(self, value):
        if self['acq']['method_mode'] == 'botorch':
            self['acq']['options'] = {'bounds': self['eval']['bounds'],
                                      'fn': self['acq']['fn'],
                                      'method': self['acq']['method'],
                                      'num_restarts': 20,
                                      'raw_samples': 20000,
                                      'spray_samples': 10,
                                      'perturb_train': False,
                                      'options':  {'maxiter': 500}}
        else:
            raise NotImplementedError


    # can't do the same for model because it requires train_targets, so this function needs to be
    # called explicitly after initial train data is available.
    def load_model_options(self, train_inputs=None, train_targets=None):
        dim = self.eval_dim
        max_radius = self.eval_max_radius
        mean_mode = self.model_mean_mode
        base_kernel = self.model_base_kernel
        base_kernel_mode = self.model_base_kernel_mode
        kernel = self.model_kernel
        kernel_mode = self.model_kernel_mode
        fixed_model_noise = 1e-3
        # Initialize outputscale and noise as in hypersphere
        outputscale_init_value = 1. if train_targets is None else train_targets.std().item()
        noise_init_value = outputscale_init_value / 1000 if fixed_model_noise is None else fixed_model_noise
        # Use same bounds as in hypersphere: exp(-12) aprx 6e-6, exp(20) aprx 5e8
        # noise (variance)
        nv_lb, nv_ub = math.exp(-12), math.exp(16),
        # lengthscales
        ls_lb, ls_ub = math.exp(-12), 2 * max_radius
        # outputscale (signal variance)
        sv_lb, sv_ub = math.exp(-12), math.exp(20)
        # Other bounds
        # ratio (nonstationary kernel)
        r_lb, r_ub = 1e-6, 1 + 1e-6

        noise_prior = UniformPrior(nv_lb, nv_ub, validate_args=None)
        constant_mean_prior = UniformPrior(train_targets.min(), train_targets.max(), validate_args=False)
        lengthscale_prior = UniformPrior(ls_lb, ls_ub, validate_args=False)
        outputscale_prior = UniformPrior(sv_lb, sv_ub, validate_args=False)


        ard_num_dims = dim if base_kernel_mode == 'ard' and kernel not in ['ck', 'ick'] else None
        self['model']['options'] = {  # observation noise, if ground truth is known
                                    'dim': dim,
                                    'anchor': self.prior_input,
                                    'likelihood_options': {'fixed_noise': fixed_model_noise,
                                                           'noise_prior': noise_prior if fixed_model_noise is None\
                                                                          else None,
                                                           'noise_constraint': interval(nv_lb, nv_ub,
                                                                                        initial_value=noise_init_value),
                                                           },
                                    'mean_options': {'mode': mean_mode,
                                                     'constant_prior': constant_mean_prior},

                                    'kernel_options': {'ard': True,
                                                       'ard_num_dims': ard_num_dims,
                                                       'dim': dim,
                                                       'kernel': kernel,
                                                       'base_kernel': base_kernel,
                                                       'mode': kernel_mode,
                                                       'lengthscale_prior': lengthscale_prior,
                                                       'lengthscale_constraint': interval(ls_lb, ls_ub,
                                                                                          initial_value=max_radius),
                                                       'outputscale_prior': outputscale_prior,
                                                       'outputscale_constraint': interval(sv_lb, sv_ub,
                                                                                  initial_value=outputscale_init_value),

                                                       }


                                    }

        if mean_mode.startswith('quad'):
            self['model']['options']['mean_options']['mode'] = 'quad'
            a_prior_modes = {'': None,
                             'hs': HalfHorseshoePrior(scale=2, validate_args=False),
                             'u':  UniformPrior(sv_lb, sv_ub, validate_args=False),
                             'ln': LogNormalPrior(loc=0, scale=2, validate_args=False)}
            a_prior_mode = mean_mode.replace('quad', '')
            greedy_anchor = True
            if a_prior_mode.startswith('f'):
                greedy_anchor = False
                a_prior_mode = a_prior_mode[1:]

            a_mean_prior = a_prior_modes[a_prior_mode]
            self['model']['options']['mean_options'].update({'greedy_anchor': greedy_anchor,
                                                             'a_prior': a_mean_prior,
                                                             'a_constraint': Positive()
                                                             })
        if kernel == 'ck':
            valid_kernel_modes = ['base']
            self.raise_error_if(kernel_mode not in valid_kernel_modes, error=NotImplementedError,
                                msg=f'For kernel={kernel}, kernel_mode={kernel_mode} is not implemented.'
                                    f'The list of valid kernel modes is {valid_kernel_modes}.')
            original = True
            if original:
                gpytorch.settings.lazily_evaluate_kernels._state = False
            self['model']['options']['kernel_options'].update({'original': original,
                                             'num_angular_weights': 1 + 3,  # up to max_power 3
                                             'angular_weights_constraint': interval_exp(sv_lb, sv_ub, initial_value=1),
                                             'angular_weights_prior':
                                                 LogNormalPrior(loc=0, scale=2, validate_args=False),
                                             'warping': True,
                                             'alpha_constraint': interval_exp(0.5, r_ub, initial_value=1),
                                             'alpha_prior':
                                                 LogSpikeUniformSlabPrior(loc=np.log(1.), low=np.log(0.5),
                                                                          high=np.log(r_ub), spike_scale=0.01,
                                                                          validate_args=False),
                                                 #SpikeUniformSlabPrior(loc=1, low=0.5, high=1+1e-9, spike_scale=0.01,
                                                 #                      validate_args=False) if warp else None,
                                             'beta_constraint': interval_exp(1 - 1e-6, 1 + r_ub, initial_value=1),
                                             'beta_prior':
                                                 LogSpikeUniformSlabPrior(loc=np.log(1.), low=np.log(1. - 1e-6),
                                                                          high=np.log(2), spike_scale=0.01,
                                                                          validate_args=False)
                                                 #SpikeUniformSlabPrior(loc=1, low=1 - 1e-6, high=2, spike_scale=0.01,
                                                 #                      validate_args=False) if warp else None
                                             })
        elif kernel == 'ik':
            valid_kernel_modes_base = ['fixed', 'greedy']
            valid_kernel_modes = self.to_list(self.model_kernel_mode_choices,
                                              filter=valid_kernel_modes_base,
                                              filter_operator='startswith')
            self.raise_error_if(kernel_mode not in valid_kernel_modes, error=NotImplementedError,
                                msg=f'For kernel={kernel}, kernel_mode={kernel_mode} is not implemented.'
                                    f'The list of valid kernel modes is {valid_kernel_modes}.')
            pool = self.model_pool
            has_nsv = self.model_has_nsv
            nslv = self.model_nslv
            nsrv = self.model_nsrv
            has_nsl = self.model_has_nsl
            nsll = self.model_nsll
            nsrl = self.model_nsrl

            nslv, lengthscale_variance_requires_grad, lengthscale_variance_prior = \
                _parse_nonstationary_lengthscale(nslv, has_nsv, ls_lb, ls_ub, max_radius)
            nsll, lengthscale_lengthscale_requires_grad, lengthscale_lengthscale_prior = \
                _parse_nonstationary_lengthscale(nsll, has_nsl, ls_lb, ls_ub, max_radius)

            nsrv, ratio_variance_requires_grad, ratio_variance_prior = _parse_nonstationary_ratio(nsrv)
            nsrl, ratio_lengthscale_requires_grad, ratio_lengthscale_prior = _parse_nonstationary_ratio(nsrl)

            self['model']['options']['kernel_options'].update({
                                  'pool': True,
                                  'requires_grad': False,
                                  'has_nonstationary_variance': has_nsv,
                                  'lengthscale_variance': nslv,
                                  'lengthscale_variance_constraint': interval(ls_lb, ls_ub, initial_value=nslv)\
                                                                     if nslv not in ['e', 's'] else None,
                                  'lengthscale_variance_requires_grad': lengthscale_variance_requires_grad,
                                  'lengthscale_variance_prior': lengthscale_variance_prior,
                                  'ratio_variance': nsrv,
                                  'ratio_variance_constraint': interval(r_lb, r_ub, initial_value=nsrv),
                                  'ratio_variance_requires_grad': ratio_variance_requires_grad,
                                  'ratio_variance_prior': ratio_variance_prior,

                                  'has_nonstationary_lengthscale': has_nsl,
                                  'lengthscale_lengthscale': nsll,
                                  'lengthscale_lengthscale_constraint': interval(ls_lb, ls_ub, initial_value=nsll)\
                                                                        if nsll not in ['e', 's'] else None,
                                  'lengthscale_lengthscale_requires_grad': lengthscale_lengthscale_requires_grad,
                                  'lengthscale_lengthscale_prior': lengthscale_lengthscale_prior,
                                  'ratio_lengthscale': nsrl,
                                  'ratio_lengthscale_constraint': interval(r_lb, r_ub, initial_value=nsrl),
                                  'ratio_lengthscale_requires_grad': ratio_lengthscale_requires_grad,
                                  'ratio_lengthscale_prior': ratio_lengthscale_prior
                                  })
        print(self['model']['options'])
        self['model']['cls'] = ExactGPModel
        return {'cls': self.model_cls, 'options': self.model_options}

def _parse_nonstationary_lengthscale(nsl, has_param, ls_lb, ls_ub, max_radius):
    lengthscale_requires_grad = False
    lengthscale_prior = None
    if nsl is not None and isinstance(nsl, str):
        if nsl not in ['e', 's']:
            lengthscale_requires_grad = True
            if nsl.startswith('u'):
                lengthscale_prior = UniformPrior(ls_lb, ls_ub, validate_args=False)
            else:
                raise NotImplementedError
            digits = str(Caster.only_digits(nsl))
            nsl = float(digits[0] + '.' + digits[1:] if digits.startswith('0') else digits)
    # NOTE: lengthscales are scaled by factor prop to sqrt(num_dim) as given by max_radius
    if not isinstance(nsl, str):
        nsl = nsl * max_radius if has_param else None
    return nsl, lengthscale_requires_grad, lengthscale_prior

def _parse_nonstationary_ratio(nsr):
    ratio_requires_grad = False
    ratio_prior = None
    if nsr is not None and isinstance(nsr, str):
        ratio_requires_grad = True
        ratio_prior = str(Caster.only_alpha(nsr))
        if ratio_prior == 'k':
            # weakly informative, peak @ 0.1
            ratio_prior = KumaraswamyPrior(1.467, 10, validate_args=False)
        elif ratio_prior == 'ki':
            # more informative/narrow, peak @ 0.1
            ratio_prior = KumaraswamyPrior(2.253, 100, validate_args=False)
        elif ratio_prior == 'kii':
            # even more informative, peak @ 0.1
            ratio_prior = KumaraswamyPrior(3.164, 1000, validate_args=False)
        elif ratio_prior == 'u':
            ratio_prior = UniformPrior(1e-6, 1 + 1e-6, validate_args=False)
        else:
            raise NotImplementedError
        digits = str(Caster.only_digits(nsr))
        nsr = float(digits[0] + '.' + digits[1:] if digits.startswith('0') else digits)
    return nsr, ratio_requires_grad, ratio_prior

if __name__ == '__main__':
    f = BoTorchSettings()
    f.packages_initialize()
    f.argload()
    print(f.load_model_options())
    print()
    d = f.dump()
    f2 = BoTorchSettings.load(d)
    print(f2)
    print()
    print(f2.model_cls)
    print(f2.model_options)

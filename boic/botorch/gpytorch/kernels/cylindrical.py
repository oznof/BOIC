import math
from typing import Union, Optional

import numpy as np
import torch

import gpytorch
from gpytorch import settings
from gpytorch.kernels import CylindricalKernel as CK
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Interval
from gpytorch.constraints.constraints import softplus, inv_softplus
from gpytorch.priors import UniformPrior

from ..priors import LogNormalPrior, LogSpikeUniformSlabPrior

LS_LB = math.exp(-12)  # LS_UB = 2 * max_radius
SV_LB = math.exp(-12)
SV_UB = math.exp(20)

interval = lambda *args, **kwargs: Interval(*args, transform=softplus, inv_transform=inv_softplus, **kwargs)
interval_exp = lambda *args, **kwargs: Interval(*args, transform=torch.exp, inv_transform=torch.log, **kwargs)

# added defaults
# added support for max_radius as in the original implementation
# added warping mode
# added original angular weights normalization in forward (polar_forward)
# added original training and prediction
# NOTE: original implementation relies on custom procedures for mll training and prediction, the origin  must be handled
# separately. In gpytorch, this is circumvented by jittering the origin point, but by
# doing so points with small radii can differ substantially from the origin because the angular information is different
# possibly resulting in overexploration of points with small radii.
class CylindricalKernel(CK):

    def __init__(self, *args, max_radius: Union[float, torch.Tensor] = None,
                 warping: Optional[bool] = True, original: Optional[bool] = True, **kwargs):
        if original:
            gpytorch.settings.lazily_evaluate_kernels._state = False
        if max_radius is None:
            if 'dim' in kwargs:
                max_radius = float(kwargs['dim']) ** 0.5
            else:
                raise ValueError('Must set max_radius.')
        if not torch.is_tensor(max_radius):
            max_radius = torch.tensor(max_radius)
        if kwargs.get('radial_base_kernel') is None:
            LS_UB = 2 * float(max_radius)
            lengthscale_constraint = kwargs.get('lengthscale_constraint',
                                                interval(LS_LB, LS_UB, initial_value=max_radius))
            lengthscale_prior = kwargs.get('lengthscale_prior', UniformPrior(LS_LB, LS_UB, validate_args=False))
            outputscale_constraint = kwargs.get('outputscale_constraint', interval(SV_LB, SV_UB,
                                                                                   initial_value=1))
            outputscale_prior = UniformPrior(SV_LB, SV_UB, validate_args=False)
            radial_base_kernel = ScaleKernel(MaternKernel(lengthscale_constraint=lengthscale_constraint,
                                                          lengthscale_prior=lengthscale_prior),
                                             outputscale_constraint=outputscale_constraint,
                                             outputscale_prior=outputscale_prior)
            kwargs['radial_base_kernel'] = radial_base_kernel
        kwargs['radial_base_kernel'].active_dims = None
        kwargs.setdefault('num_angular_weights', 4)
        kwargs.setdefault('angular_weights_constraint', interval_exp(SV_LB, SV_UB, initial_value=1))
        kwargs.setdefault('angular_weights_prior', LogNormalPrior(loc=0, scale=2, validate_args=False))
        kwargs.setdefault('alpha_constraint', interval_exp(0.5, 1 + 1e-6, initial_value=1))
        kwargs.setdefault('alpha_prior', LogSpikeUniformSlabPrior(loc=np.log(1.), low=np.log(0.5),
                                                                  high=np.log(1 + 1e-6), spike_scale=0.01,
                                                                  validate_args=False))
        kwargs.setdefault('beta_constraint', interval_exp(1 - 1e-6, 2 + 1e-6, initial_value=1))
        kwargs.setdefault('beta_prior', LogSpikeUniformSlabPrior(loc=np.log(1.), low=np.log(1. - 1e-6),
                                                                 high=np.log(2), spike_scale=0.01,
                                                                 validate_args=False))
        super().__init__(*args, **kwargs)
        self.register_buffer('max_radius', max_radius)
        self.warping = warping
        if not self.warping:
            self.raw_alpha.requires_grad = False
            self.raw_beta.requires_grad = False
        self.original = original  # if original, uses custom methods for training and prediction based
                                  # on the original implementation (HyperSphere)

    @property
    def base_kernel(self):
        return self.radial_base_kernel

    @property
    def stationary_lengthscale_kernel(self):
        kernel = self.base_kernel
        if isinstance(kernel, ScaleKernel):
            kernel = kernel.base_kernel
        return kernel

    @property
    def stationary_lengthscale(self):
        return self.stationary_lengthscale_kernel.lengthscale

    @stationary_lengthscale.setter
    def stationary_lengthscale(self, value):
        self.stationary_lengthscale_kernel.lengthscale = value

    @property
    def raw_stationary_lengthscale(self):
        return self.stationary_lengthscale_kernel.raw_lengthscale

    @property
    def raw_stationary_lengthscale_constraint(self):
        return self.stationary_lengthscale_kernel.raw_lengthscale_constraint

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: Optional[bool] = False,
                jitter: Optional[bool] = True, **kwargs) -> torch.Tensor:
        if jitter:
            return self.jitter_forward(x1, x2, diag, **kwargs)
        return self.test_forward(x1, x2, diag, **kwargs)

    def polar_forward(self, r1, a1, r2, a2, diag, **kwargs):
        if torch.any(r1 > self.max_radius + 1e-6) or torch.any(r2 > self.max_radius + 1e-6):
            raise RuntimeError(f"Cylindrical kernel not defined for data points with radius > {float(self.max_radius)}.")
        # NOTE: original implementation normalizes angular weights
        angular_weights = self.angular_weights
        angular_weights = angular_weights.div(angular_weights.sum(dim=-1, keepdim=True))
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = angular_weights[..., 0, None, None]
                else:
                    angular_kernel = angular_kernel + angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = angular_weights[..., 0, None]
                else:
                    angular_kernel = angular_kernel + angular_weights[..., p, None].mul(gram_mat.pow(p))
        with settings.lazily_evaluate_kernels(False):
            if self.warping:
                r1 = self.kuma(r1)
                r2 = self.kuma(r2)
            radial_kernel = self.radial_base_kernel(r1, r2, diag=diag, **kwargs)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)
        radii = (x / self.max_radius).clamp(min=0, max=1)
        res = self.max_radius * (1 - (1 - radii.pow(alpha) + self.eps) ** beta)
        return res

    def jitter_forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: Optional[bool] = False,
                       **kwargs) -> torch.Tensor:
        x1_, x2_ = x1.clone(), x2.clone()
        # Jitter datapoints that are exactly 0
        x1_[x1_ == 0], x2_[x2_ == 0] = x1_[x1_ == 0] + self.eps, x2_[x2_ == 0] + self.eps
        r1 = torch.sqrt(torch.sum(x1.pow(2), dim=-1, keepdim=True))
        r2 = torch.sqrt(torch.sum(x2.pow(2), dim=-1, keepdim=True))
        #r1, r2 = x1_.norm(dim=-1, keepdim=True), x2_.norm(dim=-1, keepdim=True)
        a1, a2 = x1.div(r1), x2.div(r2)
        return self.polar_forward(r1=r1, a1=a1, r2=r2, a2=a2, diag=diag, **kwargs)

    def test_forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                     xtest: Optional[torch.Tensor] = None, **kwargs):
        # In test forward, we avoid advanced indexing with boolean masks to speedup computations
        # we assume there is no origin point, unless xtest is provided. In this case
        # x2 is ignored, and we have r2 = 0 and a2 = a_xtest
        #r1 = x1.norm(dim=-1, keepdim=True)
        r1 = torch.sqrt(torch.sum(x1.pow(2), dim=-1, keepdim=True))
        a1 = x1.div(r1)
        if isinstance(xtest, torch.Tensor):
            r2 = torch.zeros(xtest.shape[:-1] + (1,), device=x1.device)
            #xtest_norm = xtest.norm(dim=-1, keepdim=True)
            xtest_norm = torch.sqrt(torch.sum(xtest.pow(2), dim=-1, keepdim=True))
            assert (torch.all(xtest_norm > 0))  # just for safety
            a2 = xtest.div(xtest_norm)
        else:
            r2 = torch.sqrt(torch.sum(x2.pow(2), dim=-1, keepdim=True))
            #r2 = x2.norm(dim=-1, keepdim=True)
            a2 = x2.div(r2)
        return self.polar_forward(r1=r1, a1=a1, r2=r2, a2=a2, diag=diag, **kwargs)

def _original_training_strategy_cylindrical(model, save_cache=True):
    # NOTE: returns negative marginal log likelihood
    x_train = model.train_inputs
    if isinstance(x_train, tuple):
        x_train = x_train[0]
    y_train = model.train_targets
    mask_train_0 = torch.all(x_train == 0, dim=-1)
    mask_train_not0 = ~mask_train_0
    x_train_not0 = x_train[mask_train_not0]
    x_train_0 = x_train[mask_train_0]
    eye_mat = torch.eye(x_train_not0.size(-2), dtype=x_train.dtype, device=x_train.device)
    #k_not0_not0 = model.covar_module(x_train_not0, x_train_not0, jitter=False)
    with settings.lazily_evaluate_kernels(False):
        k_not0_not0 = model.covar_module(x_train_not0, x_train_not0, jitter=False)
    if isinstance(k_not0_not0, gpytorch.lazy.LazyTensor):
        k_not0_not0 = k_not0_not0.evaluate()
    gram_mat = k_not0_not0 + (model.likelihood.noise + 1e-6 * model.covar_module.base_kernel.outputscale) * eye_mat
    if torch.any(torch.isnan(gram_mat)) or torch.any(k_not0_not0.diag() == 0):
        r = 0
        for p in model.covar_module.parameters(): # todo: check this
            if p.requires_grad:
                r = r + 0 * p.sum()
        return torch.tensor(float('nan')) + r
    chol_jitter = 0
    while True:
        try:
            l_not0_not0 = torch.linalg.cholesky(gram_mat + eye_mat * chol_jitter)
            break
        except RuntimeError:
            chol_jitter = max(1e-2, gram_mat.data[0, 0]) * 1e-6 if chol_jitter == 0 else chol_jitter * 10
    y_delta = y_train - model.mean_module(x_train)
    y_delta_not0 = y_delta[mask_train_not0]
    y_delta_0 = y_delta[mask_train_0]
    if save_cache:
        model.prediction_strategy = (x_train_not0, x_train_0, l_not0_not0, chol_jitter,
                                     y_delta, y_delta_not0, y_delta_0)
    with settings.lazily_evaluate_kernels(False):
        # k_0_0 = model.kernel(x1=x_train_0, x2=x_train_0) * (1 + 1e-6)
        # k_not0_0 = model.kernel(x1=x_train_not0, xtest=x_train_not0, ignore_x2=True)
        k_0_0 = model.covar_module.base_kernel.outputscale * (1 + 1e-6)
        k_not0_0 = model.covar_module(x1=x_train_not0, xtest=x_train_not0, jitter=False)
    if isinstance(k_not0_0, gpytorch.lazy.LazyTensor):
        k_not0_0 = k_not0_0.evaluate()
    chol_solver = torch.linalg.solve(l_not0_not0, torch.cat([y_delta_not0[:, None], k_not0_0], 1))
    chol_solver_y = chol_solver[:, :1]  # L_11^-1 y1  (n_1 x 1)
    chol_solver_q = chol_solver[:, 1:]  # L_10 = L_11^-1 K_10  (n_1 x n_1)
    sol_p_sqr = k_0_0 + model.likelihood.noise + chol_jitter - (chol_solver_q ** 2).sum(0).view(-1, 1)
    sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))  # L_00  normally (1 x 1), but expanded to (n_1, 1)
    # solve related to the origin.
    # L_00^-1 y_0 - L_00^-1 L_01 L_11^-1 y1
    # since L_00, L_01 have been expanded -> (n1 x 1)
    # these are then averaged out in nll
    sol_y_i = (y_delta_0 - chol_solver_q.t().mm(chol_solver_y)) / sol_p
    nll = 0.5 * (torch.sum(chol_solver_y ** 2) + torch.mean(sol_y_i ** 2)) + torch.sum(
        torch.log(torch.diag(l_not0_not0))) + torch.mean(torch.log(sol_p)) + 0.5 * x_train.size(0) * \
          math.log(2 * math.pi)
    return nll

# extended to compute covariances, but disabled by default because we are often interested only in
# univariate marginals
def _original_prediction_strategy_cylindrical(model, x_pred: torch.Tensor,
                                              compute_posterior_covariance=False):
    # NOTE: assuming x_train and x_pred to be 2D matrices, just like in the original method (does not support batches)
    if not hasattr(model, 'prediction_strategy') or not isinstance(model.prediction_strategy, tuple):
        # use prediction_strategy to store l_not0_not0, and other auxiliary variables related to training data
        # only when set_data is called, this attribute will be set to None
        x_train = model.train_inputs
        if isinstance(x_train, tuple):
            x_train = x_train[0]
        mask_train_0 = torch.all(x_train == 0, dim=-1)
        mask_train_not0 = ~mask_train_0
        x_train_not0 = x_train[mask_train_not0]
        x_train_0 = x_train[mask_train_0]
        eye_mat = torch.eye(x_train_not0.size(-2), dtype=x_train.dtype, device=x_train.device)
        with gpytorch.settings.lazily_evaluate_kernels(False):
            gram_mat = model.covar_module(x_train_not0, x_train_not0, jitter=False) + \
                       (model.likelihood.noise + 1e-6 * model.covar_module.base_kernel.outputscale) * eye_mat
        if isinstance(gram_mat, gpytorch.lazy.LazyTensor):
            gram_mat = gram_mat.evaluate()
        chol_jitter = 0
        while True:
            try:
                l_not0_not0 = torch.linalg.cholesky(gram_mat + eye_mat * chol_jitter)
                break
            except RuntimeError:
                chol_jitter = max(1e-2, gram_mat.data[0, 0]) * 1e-6 if chol_jitter == 0 else chol_jitter * 10
        y_delta = model.train_targets - model.mean_module(x_train)
        y_delta_not0 = y_delta[mask_train_not0]
        y_delta_0 = y_delta[mask_train_0]
        model.prediction_strategy = (x_train_not0, x_train_0, l_not0_not0, chol_jitter,
                                     y_delta, y_delta_not0, y_delta_0)
    (x_train_not0, x_train_0, l_not0_not0, chol_jitter, y_delta, y_delta_not0, y_delta_0) = model.prediction_strategy

    *batch_shape, n_pred, dim = x_pred.shape
    # x_pred_radius = torch.sqrt(torch.sum(x_pred ** 2, 1, keepdim=True))
    # assert((x_pred_radius.detach() > 0).all())
    with gpytorch.settings.lazily_evaluate_kernels(False):
        # k_0_0 = model.kernel(x1=x_train_0, x2=x_train_0) * (1 + 1e-6)
        k_0_0 = model.covar_module.base_kernel.outputscale * (1 + 1e-6)
        k_not0_0 = model.covar_module(x1=x_train_not0, xtest=x_pred, jitter=False)
        k_not0_pre = model.covar_module(x1=x_train_not0, x2=x_pred, jitter=False)
        #r_pred = x_pred.norm(dim=-1, keepdim=True)#.view(-1, 1)  # batches, n, 1
        r_pred = torch.sqrt(torch.sum(x_pred.pow(2), dim=-1, keepdim=True))
        if model.covar_module.warping:
            r_pred = model.covar_module.kuma(r_pred)
        k_pre_0 = model.covar_module.radial_base_kernel(x1=r_pred, x2=torch.zeros(1, 1, dtype=x_pred.dtype,
                                                                                  device=x_pred.device))
    if isinstance(k_not0_0, gpytorch.lazy.LazyTensor):
        k_not0_0 = k_not0_0.evaluate()
    if isinstance(k_not0_pre, gpytorch.lazy.LazyTensor):
        k_not0_pre = k_not0_pre.evaluate()
    if isinstance(k_pre_0, gpytorch.lazy.LazyTensor):
        k_pre_0 = k_pre_0.evaluate()

    y_delta_not0 = y_delta_not0[:, None]
    chol_solver = torch.linalg.solve(l_not0_not0.expand(*batch_shape, *l_not0_not0.shape),
                                     torch.cat([k_not0_0, k_not0_pre,
                                                y_delta_not0.expand(*batch_shape, *y_delta_not0.shape)], -1))
    chol_solver_q = chol_solver[..., :n_pred]  # L_10 = L_11^-1 K_10  (n_1 x n_pred), expanding origin to n_pred
    chol_solver_k = chol_solver[..., n_pred:n_pred * 2]  # L_11^-1 K_1*  (n_1 x n_pred)
    chol_solver_y = chol_solver[..., n_pred * 2:n_pred * 2 + 1]  # L_11^-1 y1  (n_1 x 1)
    sol_p_sqr = k_0_0 + model.likelihood.noise + chol_jitter - (chol_solver_q.pow(2)).sum(dim=-2, keepdim=True)\
                                                                                     .transpose(-2, -1)#.view(-1, 1)
    sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))  # L_00  normally (1 x 1), but expanded to (n_pred, 1)
    # sol_k_bar = K_*0 L_00^-1 + K_*1 L_10^-1
    # with L_10^-1 = - L_11^-T L_10 L_00^-T
    # equivalent to (less efficient):
    # sol_k_bar = k_pre_0 / sol_p
    # for i in range(n_pred):
    #    sol_k_bar[i] = sol_k_bar[i] - chol_solver_k.t()[i, :] @ chol_solver_q[:, i] / sol_p[i]
    # or sol_k_bar3 = k_pre_0 / sol_p - torch.diag(chol_solver_k.t() @ chol_solver_q).view(-1, 1) / sol_p
    sol_k_bar = (k_pre_0 - (chol_solver_q.mul(chol_solver_k)).sum(dim=-2, keepdim=True)
                                                             .transpose(-2, -1)).div(sol_p)
    # rhs solve: L_00^-1 y0 - L_00^-1 L_01 L_11^-1 y1
    sol_y_bar = (y_delta_0 - chol_solver_q.transpose(-2, -1).matmul(chol_solver_y)).div(sol_p)
    # pred_mean = mean_** + K_*1 L_11^-T L_11^-1 y1 + terms involving 0
    prior_mean =  model.mean_module(x_pred)
    correction_mean = (chol_solver_k.transpose(-2, -1).matmul(chol_solver_y) + sol_k_bar.mul(sol_y_bar)).squeeze(-1)
    #if prior_mean.shape != correction_mean.shape:
    #    prior_mean = prior_mean.unsqueeze(-1)
    pred_mean = prior_mean + correction_mean
    # pred_var = k_pre_pre_diag - ((chol_solver_k ** 2).sum(0).view(-1, 1) + sol_k_bar ** 2).squeeze()
    # K_** - K_*1 L_11^-T L_11 ^-1 K_1* -  (K_*0 L_00^-1 + K_*1 L_10^-1) ( K_*0 L_00^-1 + K_*1 L_10^-1)^T
    if settings.skip_posterior_variances.on():
        pred_covar = gpytorch.lazy.ZeroLazyTensor(*batch_shape, n_pred, n_pred)
    else:
        if not compute_posterior_covariance:
            #with settings.lazily_evaluate_kernels(False):
            #    k_pre_pre = model.covar_module(x1=x_pred, x2=x_pred, diag=True, jittter=False)
            k_pre_pre = model.covar_module.base_kernel.outputscale * (1 + 1e-6)
            term2 = - (chol_solver_k ** 2).sum(dim=-2, keepdim=True).transpose(-2, -1) - (sol_k_bar ** 2)
            pred_var = k_pre_pre + term2
            pred_var = pred_var.clamp(min=1e-12)
            pred_covar = pred_var * torch.eye(pred_var.size(-2), dtype=pred_var.dtype, device=pred_var.device)
            if settings.lazily_evaluate_kernels.on():
                pred_covar = gpytorch.lazy.lazify(pred_covar)
        else:
            k_pre_pre = model.covar_module(x1=x_pred, x2=x_pred, jitter=False)
            term1_stabilizer = 1e-6 * model.covar_module.base_kernel.outputscale * torch.eye(n_pred,
                                                                                              device=x_pred.device)
            term2 = -chol_solver_k.transpose(-2, -1).matmul(chol_solver_k) - sol_k_bar.matmul(sol_k_bar.transpose(-2, -1))
            term2 = term2 + term1_stabilizer  # we add this here because k_pre_pre might be a LazyTensor
            if settings.lazily_evaluate_kernels.on():
                term2 = gpytorch.lazify(term2)
            pred_covar = k_pre_pre + term2
            pred_covar_diag = pred_covar.diag()
            if torch.any(pred_covar_diag < 0):
                if settings.lazily_evaluate_kernels.on():
                    # NOTE: if it is batched, it will it break here
                    pred_covar.add_diag(-pred_covar_diag + pred_covar_diag.clamp(min=1e-12))
                else:
                    idx = torch.arange(n_pred, dtype=torch.long, device=pred_covar.device)
                    pred_covar[..., idx, idx] = pred_covar[..., idx, idx].clamp(min=1e-12)
    return MultivariateNormal(pred_mean, pred_covar)
import numpy as np

import torch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints.constraints import softplus, inv_softplus
from gpytorch.priors import UniformPrior

from .kernels import MaternKernel
from ..constraints import Positive


class InformativeKernel(gpytorch.kernels.kernel.Kernel):
    r"""Computes the covariance matrix between inputs :math:`\mathbf{x}_1` and :math:`\mathbf{x}_2`
    based on the Informative Covariance (https://arxiv.org/abs/2208.02704), which according to Equation 14 is
    computed as

    .. math::
        \begin{equation*}
          k_{\mathrm{I}}(\mathbf{x}_1, \mathbf{x}_2) = \sigma_0^2(\mathbf{x}_1, \mathbf{x}_2)
          k_{\mathrm{S}}(h_\lambda(\mathbf{x}_1), h_\lambda(\mathbf{x}_2)).
        \end{equation*}

    For ``variance_upweight=True``,

    .. math::
        \begin{equation*}
        \sigma_0^2(\mathbf{x}_1, \mathbf{x}_2) = \sigma_p^2 \sqrt{1 + \left(\frac{1}{r_\sigma} - 1\right )
                                                                    k_\sigma(d_\sigma(\mathbf{x}_1, \mathbf{x}_0))))}
                                                            \sqrt{1 + \left(\frac{1}{r_\sigma} - 1\right )
                                                                    k_\sigma(d_\sigma(\mathbf{x}_2, \mathbf{x}_0))))},
        \end{equation*}

   with anchor :math:`\mathbf{x}_0`, squared exponential :math:`k_\sigma`, ``ratio_variance`` :math:`r_\sigma`and
   weighted Euclidean distance :math:`d_\sigma` with lengthscale ``lengthscale_variance``.

   For ``lengthscale_downweight=True``,

    .. math::
        \begin{equation*}
        h_\lambda(\mathbf{x}) = u_\lambda^{-\frac{1}{2}}\boldsymbol{\Lambda}^{-\frac{1}{2}}\mathbf{x}, \quad
        u_\lambda(\mathbf{x}) = 1 + (r_\lambda - 1) k_\lambda (d_\lambda(\mathbf{x}, \mathbf{x}_0)),
        \end{equation*}

    with anchor :math:`\mathbf{x}_0`, squared exponential :math:`k_\lambda`, ``ratio_lengthscale`` :math:`r_\lambda`and
    weighted Euclidean distance :math:`d_\lambda` with lengthscale ``lengthscale_lengthscale``.

    .. note::
        If ``has_nonstationary_variance=False`` and ``has_nonstationary_lengthscale=False``, or
        ``ratio_variance=1`` and ``ratio_lengthscale=1``, this kernel is equivalent to
        the stationary ``base_kernel``.


    :param anchor: The location of the anchor :math:`\mathbf{x}_0`.
          Default: :math:`\mathbf{0}`
    :param pool: If ``True``, the parameters that govern the nonstationary effects are shared, i.e.,
                 ``lengthscale_lengthscale := lengthscale_variance`` and ``ratio_lengthscale := ratio_variance``.
        Default: ``True``
    :param requires_grad: If ``True``, the ``requires_grad`` flags that are associated with the parameters that govern
                          the nonstationary effects are enabled.
                 Default: ``False``
    :param has_nonstationary_variance: If ``True``, the signal variance, known as ``outputscale``, is spatially varying.
                              Default: ``True``
    :param variance_upweight: If ``True``, the spatially-varying signal variance is always greater or equal than the
                              stationary ``outputscale`` :math:`\sigma_p^2`, attaining the maximum value at the
                              location given by the ``anchor``.
                     Default: ``True``
    :param lengthscale_variance: The value of the lengthscale parameter in :math:`d_\sigma`.
                                 If ``S``, the lengthscale parameter is shared (as given by ``stationary_lengthscale``).
                                 If ``E``, the lengthscale parameter is a detached copy of ``stationary_lengthscale``.
                        Default: ``S``
    :type  lengthscale_variance: str, Number or torch.Tensor
    :param lengthscale_variance_constraint: Constraint to apply to ``lengthscale_variance``.
                                   Default: :obj:`~gpytorch.constraints.constraints.Positive(lower_bound=self.eps)`.
    :type lengthscale_variance_constraint: :obj:`~gpytorch.constraints.constraints.Interval`
    :param  lengthscale_variance_requires_grad: Set to ``True`` to learn the parameter by automatic differentiation.
                                       Default: ``False``
    :param lengthscale_variance_prior: Set this to apply a prior to ``lengthscale_variance``.
    :type  lengthscale_variance_prior: :obj:`~gpytorch.priors.Prior`, optional
    :param ratio_variance: The (initial) value of :math:`r_\sigma` in :math:`\sigma_0^2`.
                  Default: ``0.1``
    :param ratio_variance_constraint: Constraint to apply to ``ratio_variance``.
                             Default: :obj:`Interval(self.eps, 1 + self.eps, transform=softplus,
                                                     inv_transform=inv_softplus)`
    :type  ratio_variance_constraint: :obj:`~gpytorch.constraints.constraints.Interval`.
    :param  ratio_variance_requires_grad: Set to ``True`` to learn the parameter by automatic differentiation.
                                 Default: ``False``
    :param ratio_variance_prior: Set this to apply a prior to ``ratio_variance``.
    :type  ratio_variance_prior: :obj:`~gpytorch.priors.Prior`, optional
    :param has_nonstationary_lengthscale: If ``True``, the lengthscale vector is spatially varying.
                              Default: ``True``
    :param lengthscale_downweight: If ``True``, the spatially-varying lengthscale is always shorter or equal than
                                   ``stationary_lengthscale``, attaining the minimum value at the
                                   location given by the ``anchor``.
                          Default: ``True``
    :param lengthscale_lengthscale: The value of the lengthscale parameter in :math:`d_\lambda`.
                                 If ``S``, the lengthscale parameter is shared (as given by ``stationary_lengthscale``).
                                 If ``E``, the lengthscale parameter is a detached copy of ``stationary_lengthscale``.
                           Default: ``S``
    :type  lengthscale_lengthscale: str, Number or torch.Tensor
    :param lengthscale_lengthscale_constraint: Constraint to apply to ``lengthscale_lengthscale``.
                                      Default: :obj:`~gpytorch.constraints.constraints.Positive(lower_bound=self.eps)`.
    :type  lengthscale_lengthscale_constraint: :obj:`~gpytorch.constraints.constraints.Interval`
    :param  lengthscale_lengthscale_requires_grad: Set to ``True`` to learn the parameter by automatic differentiation.
                                          Default: ``False``
    :param lengthscale_lengthscale_prior: Set this to apply a prior to ``lengthscale_lengthscale``.
    :type  lengthscale_lengthscale_prior: :obj:`~gpytorch.priors.Prior`, optional
    :param ratio_lengthscale: The (initial) value of :math:`r_\lambda` in :math:`u_\lambda`.
                     Default: ``0.1``
    :param ratio_lengthscale_constraint: Constraint to apply to ``ratio_lengthscale``.
                                Default: :obj:`Interval(self.eps, 1 + self.eps, transform=softplus,
                                                        inv_transform=inv_softplus)`
    :type  ratio_lengthscale_constraint: :obj:`~gpytorch.constraints.constraints.Interval`.
    :param ratio_lengthscale_requires_grad: Set to ``True`` to learn the parameter by automatic differentiation.
                                    Default: ``False``
    :param ratio_lengthscale_prior: Set this to apply a prior to ``ratio_lengthscale``.
    :type  ratio_lengthscale_prior: :obj:`~gpytorch.priors.Prior`, optional

    :param base_kernel: The base stationary covariance :math:`k_{\mathrm{S}}`.
                        Additional ``args`` and ``kwargs`` are passed on to the constructor.
               Default: :obj:`~gpytorch.kernels.MaternKernel(nu=2.5)`.
    :type  base_kernel: :obj:`~gpytorch.kernels.Kernel`
    :param has_scale: If ``True``, ``ScaleKernel`` is applied to ``base_kernel``.
    :param outputscale_constraint: Constraint to apply to the outputscale parameter.
                          Default: ``None``, check :obj:`~gpytorch.kernels.ScaleKernel`
    :type  outputscale_constraint: :obj:`~gpytorch.constraints.constraints.Interval`.
    :param outputscale_prior: Set this to apply a prior to ``outputscale``.
                     Default: ``None``
    """

    def __init__(self, *args,
                 anchor=None, pool=True, requires_grad=False,
                 has_nonstationary_variance=True, variance_upweight=True,
                 lengthscale_variance='s', lengthscale_variance_constraint=None,
                 lengthscale_variance_requires_grad=False, lengthscale_variance_prior=None,
                 ratio_variance=0.1, ratio_variance_constraint=None,
                 ratio_variance_requires_grad=False, ratio_variance_prior=None,
                 has_nonstationary_lengthscale=True, lengthscale_downweight=True,
                 lengthscale_lengthscale='s', lengthscale_lengthscale_constraint=None,
                 lengthscale_lengthscale_requires_grad=False, lengthscale_lengthscale_prior=None,
                 ratio_lengthscale=0.1, ratio_lengthscale_constraint=None,
                 ratio_lengthscale_requires_grad=False, ratio_lengthscale_prior=None,
                 base_kernel=MaternKernel,
                 has_scale=True, outputscale_constraint=None, outputscale_prior=None,
                 **kwargs):
        super().__init__(has_lengthscale=False)
        if requires_grad:
            lengthscale_variance_requires_grad = True
            ratio_variance_requires_grad = True
            lengthscale_lengthscale_requires_grad = True
            ratio_lengthscale_requires_grad = True
        if pool:
            lengthscale_variance_requires_grad = lengthscale_variance_requires_grad or\
                                                 lengthscale_lengthscale_requires_grad
            ratio_variance_requires_grad = ratio_variance_requires_grad or ratio_lengthscale_requires_grad
        self.pool = pool
        self.has_scale = has_scale
        self.weight_ord = 2.
        self.weight_detach = False
        self.base_kernel = base_kernel(*args, **kwargs)
        self._kwargs = kwargs
        if self.has_scale:
            self.base_kernel = ScaleKernel(self.base_kernel, outputscale_prior=outputscale_prior,
                                           outputscale_constraint=outputscale_constraint)
        self.anchor = anchor
        batch_shape = self.batch_shape
        self.has_nonstationary_variance = has_nonstationary_variance
        self.has_nonstationary_lengthscale = has_nonstationary_lengthscale

        if self.has_nonstationary_variance:
            self.variance_upweight = variance_upweight
            self.shared_lengthscale_variance = isinstance(lengthscale_variance, str) and\
                                               lengthscale_variance.upper() == 'S'
            self.empirical_lengthscale_variance = isinstance(lengthscale_variance, str) and\
                                                  lengthscale_variance.upper() == 'E'
            if self.shared_lengthscale_variance:
                self.raw_lengthscale_variance = self.raw_stationary_lengthscale
                self.raw_lengthscale_variance_constraint = self.raw_stationary_lengthscale_constraint
            else:
                if lengthscale_variance_constraint is None:
                    lengthscale_variance_constraint = Positive(lower_bound=self.eps)
                lengthscale_shape = self.stationary_lengthscale.shape if self.empirical_lengthscale_variance \
                                    else (*batch_shape, 1, 1)
                self.register_parameter(name='raw_lengthscale_variance',
                                        parameter=torch.nn.Parameter(torch.zeros(lengthscale_shape),
                                                                     requires_grad=lengthscale_variance_requires_grad and \
                                                                                   not self.empirical_lengthscale_variance))
                self.register_constraint('raw_lengthscale_variance', lengthscale_variance_constraint)
                if lengthscale_variance_prior is not None:
                    self.register_prior('lengthscale_variance_prior', lengthscale_variance_prior,
                                        'lengthscale_variance')
            self.register_parameter(name='raw_ratio_variance',
                                    parameter=torch.nn.Parameter(torch.ones(*batch_shape, 1, 1),
                                                                 requires_grad=ratio_variance_requires_grad))
            if ratio_variance_constraint is None:
                ratio_variance_constraint = Interval(self.eps, 1 + self.eps, transform=softplus, inv_transform=inv_softplus)
                if ratio_variance_prior is None and ratio_variance_requires_grad:
                    ratio_variance_prior = UniformPrior(self.eps, 1 + self.eps, validate_args=False)
            self.register_constraint('raw_ratio_variance', ratio_variance_constraint)
            if ratio_variance_prior is not None:
                self.register_prior('ratio_variance_prior', ratio_variance_prior, 'ratio_variance')
            self.lengthscale_variance = lengthscale_variance
            self.ratio_variance = ratio_variance

        if self.has_nonstationary_lengthscale:
            self.lengthscale_downweight = lengthscale_downweight
            self.shared_lengthscale_lengthscale = isinstance(lengthscale_lengthscale, str) and\
                                                  lengthscale_lengthscale.upper() == 'S'
            self.empirical_lengthscale_lengthscale = isinstance(lengthscale_lengthscale, str) and\
                                                     lengthscale_lengthscale.upper() == 'E'
            self.shared_lengthscale_lengthscale = self.shared_lengthscale_lengthscale or\
                                                  (self.has_nonstationary_variance and self.pool and
                                                   self.shared_lengthscale_variance)
            self.empirical_lengthscale_lengthscale = self.empirical_lengthscale_lengthscale or\
                                                     (self.has_nonstationary_variance and self.pool and
                                                      self.empirical_lengthscale_variance)
            if not self.pool or not self.has_nonstationary_variance:
                if self.shared_lengthscale_lengthscale:
                    self.raw_lengthscale_lengthscale = self.raw_stationary_lengthscale
                    self.raw_lengthscale_lengthscale_constraint = self.raw_stationary_lengthscale_constraint
                else:
                    if lengthscale_lengthscale_constraint is None:
                        lengthscale_lengthscale_constraint = Positive(lower_bound=self.eps)
                    lengthscale_shape = self.stationary_lengthscale.shape if self.empirical_lengthscale_lengthscale\
                                        else (*batch_shape, 1, 1)
                    self.register_parameter(name='raw_lengthscale_lengthscale',
                                            parameter=torch.nn.Parameter(torch.zeros(lengthscale_shape),
                                                      requires_grad=lengthscale_lengthscale_requires_grad and\
                                                                    not self.empirical_lengthscale_lengthscale))
                    if lengthscale_lengthscale_prior is not None:
                        self.register_prior('lengthscale_lengthscale_prior', lengthscale_lengthscale_prior,
                                            'lengthscale_lengthscale')
                    self.register_constraint('raw_lengthscale_lengthscale', lengthscale_lengthscale_constraint)
                self.register_parameter(name='raw_ratio_lengthscale',
                                        parameter=torch.nn.Parameter(torch.ones(*batch_shape, 1, 1),
                                                                     requires_grad=ratio_lengthscale_requires_grad))
                if ratio_lengthscale_constraint is None:
                    ratio_lengthscale_constraint = Interval(self.eps, 1 + self.eps, transform=softplus, inv_transform=inv_softplus)
                    if ratio_lengthscale_prior is None and ratio_lengthscale_requires_grad:
                        ratio_lengthscale_prior = UniformPrior(self.eps, 1 + self.eps, validate_args=False)
                self.register_constraint('raw_ratio_lengthscale', ratio_lengthscale_constraint)
                if ratio_lengthscale_prior is not None:
                    self.register_prior('ratio_lengthscale_prior', ratio_lengthscale_prior, 'ratio_lengthscale')
                self.lengthscale_lengthscale = lengthscale_lengthscale
                self.ratio_lengthscale = ratio_lengthscale

    @property
    def stationary_lengthscale_kernel(self):
        kernel = self.base_kernel
        if isinstance(kernel, ScaleKernel):
            kernel = kernel.base_kernel
        return kernel

    @property
    def stationary_lengthscale(self):
        return self.stationary_lengthscale_kernel.lengthscale

    @property
    def raw_stationary_lengthscale(self):
        return self.stationary_lengthscale_kernel.raw_lengthscale

    @property
    def raw_stationary_lengthscale_constraint(self):
        return self.stationary_lengthscale_kernel.raw_lengthscale_constraint

    @property
    def is_stationary(self) -> bool:
        return self.has_nonstationary_variance or self.has_nonstationary_lengthscale

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        if value is None:
            value = 0
        if isinstance(value, int) or isinstance(value, float):
            value = float(value) * torch.ones(*self.batch_shape, 1, self._kwargs.get('dim', 1))
        value = torch.as_tensor(value)
        if len(value) == 1:
            value *= torch.ones(self._kwargs.get('dim', 1))
        self._anchor = value

    @property
    def lengthscale_variance(self):
        if self.has_nonstationary_variance:
            return self.raw_lengthscale_variance_constraint.transform(self.raw_lengthscale_variance)

    @lengthscale_variance.setter
    def lengthscale_variance(self, value):
        if self.has_nonstationary_variance and value is not None:
            self._set_lengthscale_variance(value)

    def _set_lengthscale_variance(self, value):
        if isinstance(value, str):
            if value.upper() == 'S':
                return
            if value.upper() == 'E':
                value = self.stationary_lengthscale.detach().clone()
            else:
                raise ValueError(f'Unsupported mode {value} for lengthscale_variance')
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale_variance)
        self.initialize(raw_lengthscale_variance=
                        self.raw_lengthscale_variance_constraint.inverse_transform(value))

    @property
    def lengthscale_lengthscale(self):
        if self.has_nonstationary_lengthscale:
            if self.has_nonstationary_variance and self.pool:
                return self.raw_lengthscale_variance_constraint.transform(self.raw_lengthscale_variance)
            else:
                return self.raw_lengthscale_lengthscale_constraint.transform(self.raw_lengthscale_lengthscale)

    @lengthscale_lengthscale.setter
    def lengthscale_lengthscale(self, value):
        if self.has_nonstationary_lengthscale and value is not None:
            if self.has_nonstationary_variance and self.pool:
                self._set_lengthscale_variance(value)
            else:
                self._set_lengthscale_lengthscale(value)

    def _set_lengthscale_lengthscale(self, value):
        if isinstance(value, str):
            if value.upper() == 'S':
                return
            if value.upper() == 'E':
                value = self.stationary_lengthscale.detach().clone()
            else:
                raise ValueError(f'Unsupported mode {value} for lengthscale_lengthscale')
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale_lengthscale)
        self.initialize(raw_lengthscale_lengthscale=
                        self.raw_lengthscale_lengthscale_constraint.inverse_transform(value))

    @property
    def ratio_variance(self):
        if self.has_nonstationary_variance:
            return self.raw_ratio_variance_constraint.transform(self.raw_ratio_variance)

    @ratio_variance.setter
    def ratio_variance(self, value):
        if self.has_nonstationary_variance and value is not None:
            self._set_ratio_variance(value)

    def _set_ratio_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_ratio_variance)
        self.initialize(raw_ratio_variance=
                        self.raw_ratio_variance_constraint.inverse_transform(value))

    @property
    def ratio_lengthscale(self):
        if self.has_nonstationary_lengthscale:
            if self.has_nonstationary_variance and self.pool:
                return self.raw_ratio_variance_constraint.transform(self.raw_ratio_variance)
            else:
                return self.raw_ratio_lengthscale_constraint.transform(self.raw_ratio_lengthscale)

    @ratio_lengthscale.setter
    def ratio_lengthscale(self, value):
        if self.has_nonstationary_lengthscale and value is not None:
            if self.has_nonstationary_variance and self.pool:
                self._set_ratio_variance(value)
            else:
                self._set_ratio_lengthscale(value)

    def _set_ratio_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_ratio_lengthscale)
        self.initialize(raw_ratio_lengthscale=
                        self.raw_ratio_lengthscale_constraint.inverse_transform(value))

    def update_nonstationary_lengthscales(self):
        if self.has_nonstationary_variance and self.empirical_lengthscale_variance:
            self.lengthscale_variance = 'E' # make a clone of stationary lengthscales
        if self.has_nonstationary_lengthscale and not self.pool and self.empirical_lengthscale_lengthscale:
            self.lengthscale_lengthscale = 'E'

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise NotImplementedError('last_dim_is_batch not implemented')
        if self.weight_detach and self.training:
            return self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        _ord = self.weight_ord
        is_lengthscale_lengthscale_scalar = self.has_nonstationary_lengthscale and \
                                            self.lengthscale_lengthscale.shape[-1] == 1
        is_lengthscale_variance_scalar = self.has_nonstationary_variance and self.lengthscale_variance.shape[-1] == 1
        x1_eq_x2 = torch.equal(x1, x2)
        x1_0 = x1 - self.anchor.expand_as(x1).to(x1)
        x2_0 = x1_0 if x1_eq_x2 else (x2 - self.anchor.expand_as(x2).to(x2))

        if is_lengthscale_lengthscale_scalar or is_lengthscale_variance_scalar:
            # precompute this
            if x1.shape[-1] == 1:
                d10_sq = x1_0.pow(2)
                d20_sq = x2_0.pow(2)
            else:
                d10 = torch.linalg.vector_norm(x1_0, ord=_ord, dim=-1, keepdim=True)
                d10_sq = d10.pow(2)
                d20 = d10 if x1_eq_x2 else torch.linalg.vector_norm(x2_0, ord=_ord, dim=-1, keepdim=True)
                d20_sq = d10_sq if x1_eq_x2 else d20.pow(2)

        if self.has_nonstationary_lengthscale:
            ratio_lengthscale = self.ratio_lengthscale
            lengthscale_lengthscale = self.lengthscale_lengthscale
            if is_lengthscale_lengthscale_scalar:
                exp10 = d10_sq.div(lengthscale_lengthscale.pow(2))
            else:
                exp10 = torch.linalg.vector_norm(x1_0.div(lengthscale_lengthscale), ord=_ord,
                                                 dim=-1, keepdim=True).pow(2)
            gk1 = torch.exp(-exp10.div(2))
            if x1_eq_x2:
                gk2 = gk1
            else:
                if is_lengthscale_lengthscale_scalar:
                    exp20 = d20_sq.div(lengthscale_lengthscale.pow(2))
                else:
                    exp20 = torch.linalg.vector_norm(x2_0.div(lengthscale_lengthscale), ord=_ord,
                                                     dim=-1, keepdim=True).pow(2)
                gk2 = torch.exp(-exp20.div(2))
            if self.lengthscale_downweight:  # u_lambda in Equation 15
                # spatially-varying lengthscales can only be shorter than the stationary ones
                x1 = x1.div(torch.sqrt(gk1 * (ratio_lengthscale - 1) + 1))
                x2 = x1 if x1_eq_x2 else x2.div(torch.sqrt(gk2 * (ratio_lengthscale - 1) + 1))
            else:
                weight_lengthscale = 1 / ratio_lengthscale
                x1 = x1.div(torch.sqrt(gk1 * (1 - weight_lengthscale) + weight_lengthscale))
                x2 = x1 if x1_eq_x2 else x2.div(torch.sqrt(gk2 * (1 - weight_lengthscale) + weight_lengthscale))

        covar = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        if self.has_nonstationary_variance:
            if not (self.has_nonstationary_lengthscale and
                    (self.pool or (self.empirical_lengthscale_lengthscale and self.empirical_lengthscale_variance) or
                    (self.shared_lengthscale_lengthscale and self.shared_lengthscale_variance))
                   ): # lengthscales are different so need to compute gk1 and gk2
                lengthscale_variance = self.lengthscale_variance
                if is_lengthscale_variance_scalar:
                    exp10 = d10_sq.div(lengthscale_variance.pow(2))
                else:
                    exp10 = torch.linalg.vector_norm(x1_0.div(lengthscale_variance), ord=_ord,
                                                     dim=-1, keepdim=True).pow(2)
                gk1 = torch.exp(-exp10.div(2))
                if x1_eq_x2:
                    gk2 = gk1
                else:
                    if is_lengthscale_variance_scalar:
                        exp20 = d20_sq.div(lengthscale_variance.pow(2))
                    else:
                        exp20 = torch.linalg.vector_norm(x2_0.div(lengthscale_variance), ord=_ord,
                                                         dim=-1, keepdim=True).pow(2)
                    gk2 = torch.exp(-exp20.div(2))
            if not x1_eq_x2 and not diag:
                gk2 = gk2.transpose(-1, -2)
            ratio_variance = self.ratio_variance
            if self.variance_upweight:
                weight_variance = 1 / ratio_variance
                w1_sqrt = torch.sqrt(gk1 * (weight_variance - 1) + 1)
                w2_sqrt = (w1_sqrt if diag else w1_sqrt.transpose(-1, -2)) if x1_eq_x2 else \
                           torch.sqrt(gk2 * (weight_variance - 1) + 1)
            else:
                w1_sqrt = torch.sqrt(gk1 * (1 - ratio_variance) + ratio_variance)
                w2_sqrt = (w1_sqrt if diag else w1_sqrt.transpose(-1, -2)) if x1_eq_x2 else \
                          torch.sqrt(gk2 * (1 - ratio_variance) + ratio_variance)
            factor_covar = w1_sqrt * w2_sqrt
            if diag:
                factor_covar = factor_covar.view(covar.shape)
            covar = covar * factor_covar
        return covar
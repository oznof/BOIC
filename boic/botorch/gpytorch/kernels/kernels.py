import math
import torch
import gpytorch
from gpytorch.kernels.kernel import default_postprocess_script


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()

# add use_cdist feature
class Distance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False, use_cdist=False):
        if use_cdist:
            res = torch.cdist(x1, x2, p=2)
            return res.pow(2)
        else:
            adjustment = x1.mean(-2, keepdim=True)
            x1 = x1 - adjustment
            x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
            # Compute squared distance matrix using quadratic expansion
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x1_pad = torch.ones_like(x1_norm)
            if x1_eq_x2:
                x2_norm, x2_pad = x1_norm, x1_pad
            else:
                x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
                x2_pad = torch.ones_like(x2_norm)
            x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
            x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
            res = x1_.matmul(x2_.transpose(-2, -1))
        if x1_eq_x2:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)
        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False, use_cdist=False):
        if use_cdist:
            res = torch.cdist(x1, x2, p=2)
        else:
            res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2, use_cdist=False)
            res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res


def _covar_dist(self, x1, x2, diag=False, last_dim_is_batch=False, square_dist=False,
               dist_postprocess_func=default_postprocess_script, postprocess=True,
               use_cdist=False, **params):
    r"""
    This is a helper method for computing the Euclidean distance between
    all pairs of points in x1 and x2.

    Args:
        :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
            First set of data.
        :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
            Second set of data.
        :attr:`diag` (bool):
            Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
        :attr:`last_dim_is_batch` (tuple, optional):
            Is the last dimension of the data a batch dimension or not?
        :attr:`square_dist` (bool):
            Should we square the distance matrix before returning?

    Returns:
        (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
        The shape depends on the kernel's mode
        * `diag=False`
        * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
        * `diag=True`
        * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
    """
    if last_dim_is_batch:
        x1 = x1.transpose(-1, -2).unsqueeze(-1)
        x2 = x2.transpose(-1, -2).unsqueeze(-1)

    x1_eq_x2 = torch.equal(x1, x2)
    # torch scripts expect tensors
    postprocess = torch.tensor(postprocess)
    res = None

    # Cache the Distance object or else JIT will recompile every time
    if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
        self.distance_module = Distance(dist_postprocess_func)

    if diag:
        # Special case the diagonal because we can return all zeros most of the time.
        if x1_eq_x2:
            res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            if postprocess:
                res = dist_postprocess_func(res)
            return res
        else:
            res = torch.linalg.norm((x1 - x2), ord=2, dim=-1)
            if square_dist:
                res = res.pow(2)
        if postprocess:
            res = dist_postprocess_func(res)
        return res

    elif square_dist:
        res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2, use_cdist=use_cdist)
    else:
        res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2, use_cdist=use_cdist)
    return res


class RBFKernel(gpytorch.kernels.rbf_kernel.RBFKernel):

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return self.covar_dist(x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf,
                               postprocess=True, **params)

    def covar_dist(self, *args, **kwargs):
        return _covar_dist(self, *args, **kwargs)


class MaternKernel(gpytorch.kernels.matern_kernel.MaternKernel):

    def forward(self, x1, x2, diag=False, **params):
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        return constant_component * exp_component

    def covar_dist(self, *args, **kwargs):
        return _covar_dist(self, *args, **kwargs)

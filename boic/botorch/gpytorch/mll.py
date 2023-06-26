from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood as MLL
from boic.botorch.gpytorch.kernels.cylindrical import CylindricalKernel, _original_training_strategy_cylindrical


class ExactMarginalLogLikelihood(MLL):

    def forward(self, function_dist, target, *params):
        covar_module = self.model.covar_module
        if isinstance(covar_module, CylindricalKernel) and covar_module.original:
            mll = -_original_training_strategy_cylindrical(self.model, save_cache=True)
            res = self._add_other_terms(mll, params)
            return res
        return super().forward(function_dist, target, *params)
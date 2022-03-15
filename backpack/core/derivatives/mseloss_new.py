"""NLL extention for Mean Square Error Loss."""
from abc import ABC
from backpack.core.derivatives.nll_base import NLLLossDerivatives
from torch.distributions import MultivariateNormal
from torch import tensor, zeros, eye, mul
from math import sqrt


class MSELossDerivatives(NLLLossDerivatives, ABC):
    def hessian_is_psd(self) -> bool:
        """Return whether loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True

    def _checks(self, module):
        self._check_input_dims(module)

    def _check_input_dims(self, module):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def _make_distribution(self, module, subsampling, M, N, D):
        return MultivariateNormal(zeros(D * N), eye(D * N))

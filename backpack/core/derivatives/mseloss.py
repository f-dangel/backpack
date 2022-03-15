"""NLL extention for Mean Square Error Loss."""
from abc import ABC

from torch import eye, mul, zeros
from torch.distributions import MultivariateNormal

from backpack.core.derivatives.nll_base import NLLLossDerivatives


class MSELossDerivatives(NLLLossDerivatives, ABC):
    """Partial derivatives for mean square erro loss.

    This comes from the Gaussian distribution.
    """
    def _checks(self, module):
        self._check_input_dims(module)

    def _make_distribution(self, subsampled_input, mc_samples):
        return MultivariateNormal(
            zeros(len(subsampled_input) * len(subsampled_input[0])),
            mul(eye(len(subsampled_input) * len(subsampled_input[0])), 2),
        )

    def _check_input_dims(self, module):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self) -> bool:
        return True

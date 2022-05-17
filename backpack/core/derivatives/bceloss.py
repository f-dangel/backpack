"""NLL extention for BCEWithLogits Loss."""
from typing import List, Tuple

from torch import Tensor, sigmoid
from torch.distributions import Binomial
from torch.nn import BCEWithLogitsLoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives


class BCELossDerivatives(NLLLossDerivatives):
    """Derivatives of the BCEWithLogits Loss."""

    def __init__(self, use_autograd: bool = True):
        """Initialization for BCEWithLogits loss derivative.

        Args:
            use_autograd: Compute gradients with autograd (rather than manual)
                Defaults to ``False`` (manual computation).
        """
        super().__init__(use_autograd=use_autograd)

    def _sqrt_hessian(
        self,
        module: BCEWithLogitsLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        raise NotImplementedError

    def _sum_hessian(self, module, g_inp, g_out):
        """The Hessian, summed across the batch dimension.

        Args:
            module: (torch.nn.BCEWithLogitsLoss) module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output

        Returns: a `[D, D]` tensor of the Hessian, summed across batch

        """
        raise NotImplementedError

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        raise NotImplementedError

    def _verify_support(self, module: BCEWithLogitsLoss):
        """Verification of module support for BCEWithLogitsLoss.

        Currently BCEWithLogitsLoss only supports binary output tensors,
        2D inputs, and default parameters.

        Args:
            module: (torch.nn.BCEWithLogitsLoss) module
        """
        """"""
        self._check_binary(module)
        self._check_is_default(module)
        self._check_input_dims(module)

    def _check_binary(self, module: BCEWithLogitsLoss):
        """Raises exception if outputs are not binary.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            ValueError: if outputs non-binary.
        """
        if False in [x == 0 or x == 1 for x in module.input1]:
            raise ValueError(
                "Only 0 and 1 output values are currently supported for BCEWithLogits loss."
            )

    def _check_is_default(self, module: BCEWithLogitsLoss):
        """Raises exception if module parameters are not default.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            ValueError: if module parameters non-default.
        """
        if module.weight is not None:
            raise ValueError(
                "Only None weight is currently supported for BCEWithLogits loss."
            )
        if module.reduction != "mean":
            raise ValueError(
                "Only mean reduction is currently supported for BCEWithLogits loss."
            )
        if module.pos_weight is not None:
            raise ValueError(
                "Only None pos_weight is currently supported for BCEWithLogits loss."
            )

    def _check_input_dims(self, module: BCEWithLogitsLoss):
        """Raises an exception if the shapes of the input are not supported.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            ValueError: if input not 2D.
        """
        if not len(module.input0.shape) == 2:
            raise ValueError(
                "Only 2D inputs are currently supported for BCEWithLogitsLoss."
            )

    def _make_distribution(self, subsampled_input: Tensor):
        """Make the sampling distribution for the NLL loss form of BCEWithLogits.

        The BCEWithLogitsLoss ∝ ∑ᵢ₌₁ⁿ Yᵢ log 𝜎(xᵢ) + (1 − Yᵢ) log(1− 𝜎(xᵢ)).
        The log likelihood of the Binomial distribution is
        Yᵢ log p(xᵢ) + (1 − Yᵢ) log(1 − p(xᵢ)), so these are equivalent if
        p(xᵢ) = 𝜎(xᵢ).

        Args:
            subsampled_input: input after subsampling

        Returns:
            torch.distributions Binomial distribution with probabilities equal to
                the sigmoid of subsampled_input.
        """
        return Binomial(probs=sigmoid(subsampled_input))

    def hessian_is_psd(self) -> bool:
        """Return whether BCEWithLogits loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        return input.numel()

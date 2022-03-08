from torch import einsum

from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.hbp.hbp_options import (
    BackpropStrategy,
    ExpectationApproximation,
)
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack.utils import conv as convUtils


class HBPConv2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        """
        Calculates the weight Hessian.

        Here it is calculated on its own, but the original KFAC paper calculates
        the bias Hessian along with the weight Hessian by appending ones to the
        end of the activations.

        Additional differences from the original can be found in _factor_from_sqrt
        and _weight_for_sqrt.

        Args:
            ext: extension object
            module: module layers
            g_inp:
            g_out:
            backproped: square root matrix

        Returns: weight Hessian

        """
        if module.groups != 1:
            raise NotImplementedError(
                f"groups ≠ 1 is not supported by {ext.__class__.__name__} "
                + f"(got {module.groups})."
            )

        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    # TODO: Require tests
    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = [self._factor_from_batch_average(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        """
        Calculate the weight Hessian with the approximation
        (see _factor_from_sqrt) to Omega and exact Gamma
        from the original KFAC paper.

        Also omit dividing by the cardinalitiy of Tau, as
        done in the original paper.

        Args:
            ext: extension object
            module: module layers
            backproped: square root matrix

        Returns: weight Hessian

        """
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        """
        Calculate the factors from the inputs.

        This is the equivalent to the original KFAC paper Omega
        calculation.

        Args:
            ext: extension object
            module: module layers

        Returns: Input component (Omega)

        """
        X = convUtils.unfold_input(module, module.input0)
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")
        else:
            yield einsum("bik,bjk->ij", (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        """
        Calculates the factors from the activations.

        The original KFAC paper does the same calculation for Gamma, but without
        summing across the patches first ("einsum("cbij->cbi", (sqrt_ggn,))").

        Args:
            module: module layers
            backproped: square root matrix

        Returns: activation component (approximately Gamma)

        """
        sqrt_ggn = backproped

        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum("cbij->cbi", (sqrt_ggn,))
        return einsum("cbi,cbl->il", (sqrt_ggn, sqrt_ggn))

    def bias(self, ext, module, g_inp, g_out, backproped):
        """
        Calculates the bias Hessian.

        Here it is calculated on its own, but the original KFAC paper calculates
        the bias Hessian along with the weight Hessian by appending ones to the
        end of the activations.

        Additional differences from the original can be found in _factor_from_sqrt
        and _bias_for_sqrt.

        Args:
            ext: extension object
            module: module layers
            g_inp:
            g_out:
            backproped: square root matrix

        Returns: bias Hessian

        """
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(module, backproped)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._bias_for_sqrt(module, backproped)

    def _bias_for_sqrt(self, module, backproped):
        """
        Calculate the bias Hessian with the approximation
        (see _factor_from_sqrt) to Omega from the original
        KFAC paper.

        Also omit dividing by the cardinalitiy of Tau, as
        done in the original paper.

        Args:
            module: module layers
            backproped: square root matrix

        Returns: bias Hessian

        """
        return [self._factor_from_sqrt(module, backproped)]

    # TODO: Require tests
    def _bias_for_batch_average(self, module, backproped):
        return [self._factor_from_batch_average(module, backproped)]

    def _factor_from_batch_average(self, module, backproped):
        _, out_c, out_x, out_y = module.output.size()
        out_pixels = out_x * out_y
        # sum over spatial coordinates
        result = backproped.view(out_c, out_pixels, out_c, out_pixels).sum([1, 3])
        return result.contiguous()

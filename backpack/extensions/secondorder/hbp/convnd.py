"""Kronecker approximations of the Hessian for convolution layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

from torch import Tensor, einsum
from torch.nn import Conv1d, Conv2d, Conv3d

from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.hbp.hbp_options import (
    BackpropStrategy,
    ExpectationApproximation,
)
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack.utils import conv as convUtils

if TYPE_CHECKING:
    from backpack.extensions.secondorder.hbp import HBP


class HBPConvNd(HBPBaseModule):
    """Computes Kronecker-structured Hessian approximations for convolution layers.

    NOTE docstrings use 2d convolution to explain the arguments and output shapes.
    """

    def __init__(self, N: int):
        """Store dimension of convolution.

        Args:
            N: Dimension of convolution.
        """
        self._conv_dim = N
        derivatives_cls = {
            1: Conv1DDerivatives,
            2: Conv2DDerivatives,
            3: Conv3DDerivatives,
        }[N]
        super().__init__(derivatives_cls(), params=["weight", "bias"])

    def weight(
        self,
        ext: HBP,
        module: Union[Conv1d, Conv2d, Conv3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> List[Tensor]:
        """Compute the Kronecker factors for the weight Hessian approximation.

        Note:
            TODO The Kronecker factor computed from the backpropagated quantity
            differs from the KFC paper (https://arxiv.org/pdf/1602.01407.pdf)
            by a factor of |Τ| = H * W where H, W denote the spatial output
            dimensions of the convolution. If this convention is changed to be
            more consistent with the paper, this must be clearly communicated
            to users as it will alter the scale of the KFAC quantity for weights of
            convolutions in comparison to older versions.

        Args:
            ext: HBP extension.
            module: convolution layer the backpropagation is performed on.
            g_inp: input gradient.
            g_out: output gradient.
            backproped: Backpropagated quantity, depends on the approximation mode.
                For KFLR/KFAC this is the MC/exact matrix square root of the GGN w.r.t.
                the convolution output (shape `[M, N, C, H, W]`) and has shape
                `[M, N, C, H, W]` with `M` the number of MC samples or the number of
                classes for KFAC/KFLR, respectively. For KFRA, the backpropagated
                object approximates the batch-averaged GGN w.r.t. to the convolution
                output and has shape `[C * H * W, C * H * W]`.

        Returns:
            List of Kronecker factors whose Kronecker product approximates the weight
            Hessian. Its length depends on the Hessian approximation. If `[A, B, C]`
            is returned, then `A ⊗ B ⊗ C` has shape `[weight.numel(), weight.numel()]`
            and approximates the weight Hessian.
        """
        self._maybe_raise_groups_not_implemented_error(ext, module)

        kron_factors: List[Tensor] = []
        bp_strategy = ext.get_backprop_strategy()

        # TODO Find a setting in which this corresponds to an autodiff quantity, test
        if BackpropStrategy.is_batch_average(bp_strategy):  # KFRA
            kron_factors.append(self._factor_from_batch_average(module, backproped))

        elif BackpropStrategy.is_sqrt(bp_strategy):  # KFLR, KFAC
            kron_factors.append(self._factor_from_sqrt(module, backproped))

        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(
        self, ext: HBP, module: Union[Conv1d, Conv2d, Conv3d]
    ) -> List[Tensor]:
        """Compute the un-centered covariance of the unfolded input.

        In the notation of https://arxiv.org/pdf/1602.01407.pdf,
        this computes Ω from equation (32) for KFAC.

        Args:
            ext: HBP extension.
            module: Convolution layer.

        Raises:
            NotImplementedError: If the backpropagation strategy differs from KFAC.

        Returns:
            List containing the tensor of the un-centered covariance of the unfolded
            input. For a convolution with kernel of size `[_, C_in, K_H, K_W]`, its
            shape is `[C_in * K_H * K_W, C_in * K_H * K_W]`.
        """
        ea_strategy = ext.get_ea_strategy()
        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")

        X = convUtils.unfold_input(module, module.input0)

        return [einsum("bik,bjk->ij", X, X) / X.shape[0]]

    def _factor_from_sqrt(
        self, module: Union[Conv1d, Conv2d, Conv3d], backproped: Tensor
    ) -> Tensor:
        """Compute the Kronecker factor from the backpropagated GGN matrix square root.

        In the notation of https://arxiv.org/pdf/1602.01407.pdf,
        this computes |Τ| * Γ from equation (32) for KFAC.

        Note:
            In comparison to the KFC paper, the output differs by a factor of |Τ|.
            This is because |Τ| * Γ is the MC/exact GGN w.r.t. the convolution's bias.
            For two-dimensional convolution with output of shape `[N, C_out, H, W]`,
            |Τ| = H * W.

        Args:
            module: Convolution layer.
            backproped: Backpropagated quantity, corresponding to the MC/exact matrix
                square square root of the GGN w.r.t. the convolution output. For a
                convolution with output shape `[N, C_out, H, W]`, this square root is
                of shape `[M, N, C_out, H, W]` where `M` is the number of MC samples
                for KFAC, and the number of classes for KFLR. The matrix square root
                already incorporates a normalization factor for batch size averaging.

        Returns:
            MC/exact GGN w.r.t. the bias. Has shape `[C_out, C_out]`
        """
        sqrt_ggn = backproped.flatten(start_dim=-self._conv_dim)
        sqrt_ggn = einsum("cbij->cbi", sqrt_ggn)
        return einsum("cbi,cbl->il", sqrt_ggn, sqrt_ggn)

    def bias(
        self,
        ext: HBP,
        module: Union[Conv1d, Conv2d, Conv3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> List[Tensor]:
        """Compute the Kronecker factors for the bias Hessian approximation.

        Args:
            ext: HBP extension.
            module: convolution layer the backpropagation is performed on.
            g_inp: input gradient.
            g_out: output gradient.
            backproped: Backpropagated quantity, depends on the approximation mode.
                For KFLR/KFAC this is the MC/exact matrix square root of the GGN w.r.t.
                the convolution output (shape `[M, N, C, H, W]`) and has shape
                `[M, N, C, H, W]` with `M` the number of MC samples or the number of
                classes for KFAC/KFLR, respectively. For KFRA, the backpropagated
                object approximates the batch-averaged GGN w.r.t. to the convolution
                output and has shape `[C * H * W, C * H * W]`.

        Returns:
            List of Kronecker factors whose Kronecker product approximates the bias
            Hessian. Its length depends on the Hessian approximation. If `[A, B, C]`
            is returned, then `A ⊗ B ⊗ C` has shape `[bias.numel(), bias.numel()]`
            and approximates the bias Hessian.
        """
        kron_factors: List[Tensor] = []
        bp_strategy = ext.get_backprop_strategy()

        # TODO Find a setting in which this corresponds to an autodiff quantity, test
        if BackpropStrategy.is_batch_average(bp_strategy):  # KFRA
            kron_factors.append(self._factor_from_batch_average(module, backproped))
        elif BackpropStrategy.is_sqrt(bp_strategy):  # KFAC/KFLR
            kron_factors.append(self._factor_from_sqrt(module, backproped))

        return kron_factors

    def _factor_from_batch_average(
        self, module: Union[Conv1d, Conv2d, Conv3d], backproped: Tensor
    ) -> Tensor:
        """Compute the Kronecker factor from the backpropagated output Hessian proxy.

        Note:
            TODO Currently, the Kronecker approximation that needs to be imposed on
            the backpropagated Hessian proxy to achieve a Kronecker structure of the
            weight Hessian differs from KFC (https://arxiv.org/pdf/1602.01407.pdf).
            This could be changed for this factor to be more consistent with the
            KFC approximations. If this is changed, this must be clearly communicated
            to users as it will alter the KFRA quantity for weights of
            convolutions in comparison to older versions. NOTE that this method is
            currently shared by the weights and bias terms for KFRA, but the described
            improvement would only apply to the weights, and not the bias.

        Args:
            module: Convolution layer.
            backproped: Approximation for the batch-averaged Hessian w.r.t. the output
                of the convolution layer. Has shape `[C * H * W, C * H * W]` if the
                convolution's output is of shape `[N, C, H, W]`.

        Returns:
            Kronecker factor used for approximating the weight Hessian in convolutions.
            Has shape `[C, C]` with `C` the convolution's output channels.
        """
        spatial_dim = module.output.shape[-self._conv_dim :].numel()
        out_channels = module.output.shape[-self._conv_dim - 1]

        # sum over spatial coordinates
        return (
            backproped.view(out_channels, spatial_dim, out_channels, spatial_dim)
            .sum([1, 3])
            .contiguous()
        )

    @staticmethod
    def _maybe_raise_groups_not_implemented_error(
        ext: HBP, module: Union[Conv1d, Conv2d, Conv3d]
    ):
        """Raise NotImplementedError for grouped convolution.

        Args:
            ext: HBP extension.
            module: Convolution layer.

        Raises:
            NotImplementedError: If groups ≠ 1.
        """
        if module.groups != 1:
            ext_name = ext.__class__.__name__
            raise NotImplementedError(
                f"groups ≠ 1 is not supported by {ext_name} (got {module.groups})."
            )

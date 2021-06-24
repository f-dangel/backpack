"""Contains extensions for convolution layers used by ``GGNMP``."""
from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPConvNd(GGNMPBase):
    def weight(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def weight_ggnmp(mat):
            result = self.derivatives.weight_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, result
            )

            return result

        return weight_ggnmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def bias_ggnmp(mat):
            result = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.bias_jac_t_mat_prod(module, g_inp, g_out, result)

            return result

        return bias_ggnmp


class GGNMPConv1d(GGNMPConvNd):
    """``GGNMP`` extension for ``torch.nn.Conv1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv1d`` module."""
        super().__init__(Conv1DDerivatives(), params=["bias", "weight"])


class GGNMPConv2d(GGNMPConvNd):
    """``GGNMP`` extension for ``torch.nn.Conv2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv2d`` module."""
        super().__init__(Conv2DDerivatives(), params=["bias", "weight"])


class GGNMPConv3d(GGNMPConvNd):
    """``GGNMP`` extension for ``torch.nn.Conv3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv3d`` module."""
        super().__init__(Conv3DDerivatives(), params=["bias", "weight"])

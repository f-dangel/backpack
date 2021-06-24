"""Contains extensions for transpose convolution layers used by ``GGNMP``."""
from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPConvTransposeNd(GGNMPBase):
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


class GGNMPConvTranspose1d(GGNMPConvTransposeNd):
    """``GGNMP`` extension for ``torch.nn.ConvTranspose1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose1d`` module."""
        super().__init__(ConvTranspose1DDerivatives(), params=["bias", "weight"])


class GGNMPConvTranspose2d(GGNMPConvTransposeNd):
    """``GGNMP`` extension for ``torch.nn.ConvTranspose2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose2d`` module."""
        super().__init__(ConvTranspose2DDerivatives(), params=["bias", "weight"])


class GGNMPConvTranspose3d(GGNMPConvTransposeNd):
    """``GGNMP`` extension for ``torch.nn.ConvTranspose3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose3d`` module."""
        super().__init__(ConvTranspose3DDerivatives(), params=["bias", "weight"])

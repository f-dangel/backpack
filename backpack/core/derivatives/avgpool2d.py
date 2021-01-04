"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

import torch.nn
from torch.nn import Conv2d, ConvTranspose2d

from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives
from backpack.utils.ein import eingroup


class AvgPool2DDerivatives(AvgPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=2)

    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """Use fact that average pooling can be implemented as conv."""
        _, C, H_in, W_in = module.input0.size()
        in_features = C * H_in * W_in
        _, _, H_out, W_out = module.output.size()
        out_features = C * H_out * W_out

        mat = mat.reshape(out_features * C, 1, H_out, W_out)
        jac_t_mat = self.__apply_jacobian_t_of(module, mat).reshape(
            out_features, in_features
        )
        mat_t_jac = jac_t_mat.t().reshape(in_features * C, 1, H_out, W_out)
        jac_t_mat_t_jac = self.__apply_jacobian_t_of(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def __apply_jacobian_t_of(self, module, mat):
        C_for_conv_t = 1

        conv2d_t = ConvTranspose2d(
            in_channels=C_for_conv_t,
            out_channels=C_for_conv_t,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
        ).to(module.input0.device)

        conv2d_t.weight.requires_grad = False
        avg_kernel = torch.ones_like(conv2d_t.weight) / conv2d_t.weight.numel()
        conv2d_t.weight.data = avg_kernel

        V_N_C_in = mat.size(0)
        _, _, H_in, W_in = module.input0.size()
        output_size = (V_N_C_in, C_for_conv_t, H_in, W_in)

        return conv2d_t(mat, output_size=output_size)

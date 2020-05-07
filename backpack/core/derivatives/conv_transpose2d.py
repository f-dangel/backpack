"""Partial derivatives for `torch.nn.ConvTranspose2d`."""

from torch.nn import Conv2d, ConvTranspose2d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.ein import eingroup


class ConvTranspose2DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return ConvTranspose2d

    def hessian_is_zero(self):
        return True

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, H_axis, W_axis = 1, 3, 4
        axes = [H_axis, W_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        raise NotImplementedError

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply ConvTranspose2d backward operation."""
        _, C_in, H_in, W_in = module.input0.size()
        _, C_out, H_out, W_out = module.output.size()
        H_axis = 2
        W_axis = 3

        conv2d = Conv2d(
            in_channels=C_out,
            out_channels=C_in,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
            dilation=module.dilation,
            groups=module.groups,
        ).to(module.input0.device)

        conv2d.weight.data = module.weight

        # V_N = mat.size(0)
        # output_size = (V_N, C_in, H_in, W_in)

        # jac_t_mat = (
        #     conv2d_t(mat, output_size=output_size)
        #     .narrow(H_axis, 0, H_in)
        #     .narrow(W_axis, 0, W_in)
        # )

        jac_t_mat = conv2d(mat).narrow(H_axis, 0, H_in).narrow(W_axis, 0, W_in)
        return jac_t_mat

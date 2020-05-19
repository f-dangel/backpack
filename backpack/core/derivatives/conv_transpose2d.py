"""Partial derivatives for `torch.nn.ConvTranspose2d`."""

from torch.nn import ConvTranspose2d
from torch.nn.functional import conv2d, conv_transpose2d

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
        """Unintuitive, but faster due to convolution."""
        V = mat.shape[0]
        N, C_out, H_out, W_out = module.output_shape
        _, C_in, _, _ = module.input0_shape

        mat = mat.unsqueeze(2).repeat(1, 1, C_in, 1, 1, 1)
        mat = eingroup("v,n,c,d,w,h->vnc,d,w,h", mat)

        N_axis = 0
        input = eingroup("n,c,h,w->nc,h,w", module.input0).unsqueeze(N_axis)
        input = input.repeat(1, V, 1, 1)

        assert module.stride == module.dilation

        conv2d_t = ConvTranspose2d(
            in_channels=V * N * C_in,
            out_channels=V * N * C_out * C_in,
            kernel_size=(H_out, W_out),
            stride=module.dilation,
            padding=module.padding,
            bias=False,
            dilation=module.stride,
            groups=C_in * N * V,
        ).to(module.input0.device)
        print(conv2d_t)

        print("Conv_t weight shape:", conv2d_t.weight.shape)
        print("Mat shape:", mat.shape)
        assert conv2d_t.weight.shape == mat.shape
        conv2d_t.weight.data = mat

        K_H_axis, K_W_axis = 1, 2
        _, _, K_H, K_W = module.weight.shape

        # something is wrong with the output size
        output_size = (1, V * N * C_in * C_out, K_H, K_W)
        grad_weight = conv2d_t(input, output_size=output_size).squeeze(0)

        print(grad_weight.shape)

        grad_weight = grad_weight.narrow(K_H_axis, 0, K_H).narrow(K_W_axis, 0, K_W)

        eingroup_eq = "vnio,x,y->v,{}i,o,x,y".format("" if sum_batch else "n,")
        return eingroup(
            eingroup_eq, grad_weight, dim={"v": V, "n": N, "i": C_in, "o": C_out}
        )

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply ConvTranspose2d backward operation."""
        H_axis = 2
        W_axis = 3
        H_in = module.input0.size(H_axis)
        W_in = module.input0.size(W_axis)

        return (
            conv2d(
                mat,
                module.weight,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
            .narrow(H_axis, 0, H_in)
            .narrow(W_axis, 0, W_in)
        )

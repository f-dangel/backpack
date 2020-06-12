from torch import einsum
from torch.nn import Conv3d, ConvTranspose3d
from torch.nn.functional import conv3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import conv as convUtils
from backpack.utils.ein import eingroup


class Conv3DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return Conv3d

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_by_conv(module.input0, module)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_in, D_in, H_in, W_in = module.input0.size()
        in_features = C_in * D_in * H_in * W_in
        _, C_out, D_out, H_out, W_out = module.output.size()
        out_features = C_out * D_out * H_out * W_out

        mat = mat.reshape(out_features, C_out, D_out, H_out, W_out)
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, C_out, D_out, H_out, W_out)
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,d,c,h,w->vn,d,c,h,w", mat)
        jmp_as_conv = conv3d(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.reshape_like_output(jmp_as_conv, module)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,d,h,w->vn,c,d,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply Conv3d backward operation."""
        _, C_in, D_in, H_in, W_in = module.input0.size()
        _, C_out, _, _, _ = module.output.size()
        D_axis = 2
        H_axis = 3
        W_axis = 4

        conv3d_t = ConvTranspose3d(
            in_channels=C_out,
            out_channels=C_in,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
            dilation=module.dilation,
            groups=module.groups,
        ).to(module.input0.device)

        conv3d_t.weight.data = module.weight

        V_N = mat.size(0)
        output_size = (V_N, C_in, D_in, H_in, W_in)

        jac_t_mat = (
            conv3d_t(mat, output_size=output_size)
            .narrow(D_axis, 0, D_in)
            .narrow(H_axis, 0, H_in)
            .narrow(W_axis, 0, W_in)
        )
        return jac_t_mat

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """mat has shape [V, C_out]"""
        # expand for each batch and for each channel
        N_axis, D_axis, H_axis, W_axis = 1, 3, 4, 5
        jac_mat = (
            mat.unsqueeze(N_axis).unsqueeze(D_axis).unsqueeze(H_axis).unsqueeze(W_axis)
        )

        N, _, D_out, H_out, W_out = module.output_shape
        return jac_mat.expand(-1, N, -1, D_out, H_out, W_out)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, D_axis, H_axis, W_axis = 1, 3, 4, 5
        axes = [D_axis, H_axis, W_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    # TODO: Improve performance by using conv instead of unfold

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        jac_mat = eingroup("v,o,i,d,h,w->v,o,idhw", mat)
        X = self.get_unfolded_input(module)
        jac_mat = einsum("nij,vki->vnkj", (X, jac_mat))
        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        V = mat.shape[0]
        N, C_out, _, _, _ = module.output_shape
        _, C_in, _, _, _ = module.input0_shape

        mat = eingroup("v,n,c,d,w,h->vn,c,d,w,h", mat).repeat(1, C_in, 1, 1, 1)
        C_in_axis = 1
        # a,b represent the combined/repeated dimensions
        mat = eingroup("a,b,d,w,h->ab,d,w,h", mat).unsqueeze(C_in_axis)

        N_axis = 0
        input = eingroup("n,c,d,h,w->nc,d,h,w", module.input0).unsqueeze(N_axis)
        input = input.repeat(1, V, 1, 1, 1)

        grad_weight = conv3d(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        K_D_axis, K_H_axis, K_W_axis = 1, 2, 3
        _, _, K_D, K_H, K_W = module.weight.shape
        grad_weight = (
            grad_weight.narrow(K_D_axis, 0, K_D)
            .narrow(K_H_axis, 0, K_H)
            .narrow(K_W_axis, 0, K_W)
        )

        eingroup_eq = "vnio,x,y,z->v,{}o,i,x,y,z".format("" if sum_batch else "n,")
        return eingroup(
            eingroup_eq, grad_weight, dim={"v": V, "n": N, "i": C_in, "o": C_out}
        )

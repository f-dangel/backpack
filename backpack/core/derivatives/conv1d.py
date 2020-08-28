from torch import einsum
from torch.nn.grad import _grad_input_padding
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.functional import conv_transpose1d
from torch.nn.functional import conv1d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import conv as convUtils
from backpack.utils.ein import eingroup


class Conv1DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return Conv1d

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_by_conv(module.input0, module)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_in, L_in = module.input0.size()
        in_features = C_in * L_in
        _, C_out, L_out = module.output.size()
        out_features = C_out * L_out

        mat = mat.reshape(out_features, C_out, L_out)
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, C_out, L_out)
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,l->vn,c,l", mat)
        jmp_as_conv = conv1d(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.reshape_like_output(jmp_as_conv, module)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,l->vn,c,l", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply Conv1d backward operation."""
        _, C_in, L_in = module.input0.size()
        _, C_out, _ = module.output.size()
        V_N = mat.size(0)

        input_size = (V_N, C_in, L_in)

        grad_padding = _grad_input_padding(
            grad_output=mat,
            input_size=input_size,
            stride=module.stride,
            padding=module.padding,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
        )

        jac_t_mat = conv_transpose1d(
            input=mat,
            weight=module.weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            output_padding=grad_padding,
            groups=module.groups,
            dilation=module.dilation,
        )
        return jac_t_mat

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """mat has shape [V, C_out]"""
        # expand for each batch and for each channel
        N_axis, L_axis = 1, 3
        jac_mat = mat.unsqueeze(N_axis).unsqueeze(L_axis)

        N, _, L_out = module.output_shape
        return jac_mat.expand(-1, N, -1, L_out)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, L_axis = 1, 3
        axes = [L_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    # TODO: Improve performance by using conv instead of unfold

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        jac_mat = eingroup("v,o,i,l->v,o,il", mat)
        X = self.get_unfolded_input(module)
        jac_mat = einsum("nij,vki->vnkj", (X, jac_mat))
        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        V = mat.shape[0]
        N, C_out, _ = module.output_shape
        _, C_in, _ = module.input0_shape

        mat = eingroup("v,n,c,l->vn,c,l", mat).repeat(1, C_in, 1)
        C_in_axis = 1
        # a,b represent the combined/repeated dimensions
        mat = eingroup("a,b,l->ab,l", mat).unsqueeze(C_in_axis)

        N_axis = 0
        input = eingroup("n,c,l->nc,l", module.input0).unsqueeze(N_axis)
        input = input.repeat(1, V, 1)

        grad_weight = conv1d(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        K_L_axis = 1
        _, _, K_L = module.weight.shape
        grad_weight = grad_weight.narrow(K_L_axis, 0, K_L)

        eingroup_eq = "vnio,x->v,{}o,i,x".format("" if sum_batch else "n,")
        return eingroup(
            eingroup_eq, grad_weight, dim={"v": V, "n": N, "i": C_in, "o": C_out}
        )

from torch import zeros
from torch.nn import MaxPool2d
from torch.nn.functional import max_pool2d

from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from backpack.core.derivatives.basederivatives import BaseDerivatives

from backpack.utils.einsum import eingroup


class MaxPool2DDerivatives(BaseDerivatives):
    def get_module(self):
        return MaxPool2d

    # TODO: Do not recompute but get from forward pass of module
    def get_pooling_idx(self, module):
        _, pool_idx = max_pool2d(
            module.input0,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True,
            ceil_mode=module.ceil_mode,
        )
        return pool_idx

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """

        Note: It is highly questionable whether this makes sense both
              in terms of the approximation and memory costs.
        """
        device = mat.device
        N, channels, H_in, W_in = module.input0.size()
        in_features = channels * H_in * W_in
        _, _, H_out, W_out = module.output.size()
        out_features = channels * H_out * W_out

        pool_idx = self.get_pooling_idx(module).view(N, channels, H_out * W_out)
        result = zeros(in_features, in_features, device=device)

        for b in range(N):
            idx = pool_idx[b, :, :]
            temp = zeros(in_features, out_features, device=device)
            temp.scatter_add_(1, idx, mat)
            result.scatter_add_(0, idx.t(), temp)
        return result / N

    def hessian_is_zero(self):
        return True

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        return self.view_like_output(jmp_as_pool, module)

    def __apply_jacobian_of(self, module, mat):
        V, HW_axis = mat.shape[0], 3
        pool_idx = self.__pool_idx_for_jac(module, V)
        return mat.gather(HW_axis, pool_idx)

    def __pool_idx_for_jac(self, module, V):
        """Manipulated pooling indices ready-to-use in jac(t)."""

        pool_idx = self.get_pooling_idx(module)
        V_axis = 0
        return (
            eingroup("n,c,h,w->n,c,hw", pool_idx)
            .unsqueeze(V_axis)
            .expand(V, -1, -1, -1)
        )

    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        return self.view_like_input(jmp_as_pool, module)

    def __apply_jacobian_t_of(self, module, mat):
        V = mat.shape[0]
        result = self.__zero_for_jac_t(module, V, mat.device)
        pool_idx = self.__pool_idx_for_jac(module, V)

        HW_axis = 3
        result.scatter_add_(HW_axis, pool_idx, mat)
        return result

    def __zero_for_jac_t(self, module, V, device):
        N, C_out, _, _ = module.output_shape
        _, _, H_in, W_in = module.input0.size()

        shape = (V, N, C_out, H_in * W_in)
        return zeros(shape, device=device)

from torch import zeros
from torch.nn.functional import max_pool2d

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.ein import eingroup


class MaxPool2DDerivatives(BaseDerivatives):

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

        Note:
            Need to loop over the samples, as dealing with all at once
            requires memory for `N * C² * H_in * W_in * H_out * W_out`
            elements
        """
        device = mat.device
        N, C, H_in, W_in = module.input0.size()
        _, _, H_out, W_out = module.output.size()

        in_pixels = H_in * W_in
        in_features = C * in_pixels

        pool_idx = self.get_pooling_idx(module).view(N, C, H_out * W_out)

        def sample_ea_jac_t_mat_jac_prod(n, mat):
            jac_t_mat = sample_jac_t_mat_prod(n, mat)
            mat_t_jac = jac_t_mat.t()
            jac_t_mat_t_jac = sample_jac_t_mat_prod(n, mat_t_jac)
            return jac_t_mat_t_jac.t()

        def sample_jac_t_mat_prod(n, mat):
            num_cols = mat.size(1)
            idx = pool_idx[n, :, :].unsqueeze(-1).expand(-1, -1, num_cols)

            jac_t_mat = zeros(C, H_in * W_in, num_cols, device=device)
            mat = mat.reshape(C, H_out * W_out, num_cols)

            jac_t_mat.scatter_add_(1, idx, mat)

            return jac_t_mat.reshape(in_features, num_cols)

        result = zeros(in_features, in_features, device=device)

        for n in range(N):
            result += sample_ea_jac_t_mat_jac_prod(n, mat)

        return result / N

    def hessian_is_zero(self):
        return True

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        return self.reshape_like_output(jmp_as_pool, module)

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

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        return self.reshape_like_input(jmp_as_pool, module)

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

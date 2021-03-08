from einops import rearrange
from torch import zeros
from torch.nn.functional import max_pool1d, max_pool2d, max_pool3d

from backpack.core.derivatives.basederivatives import BaseDerivatives


class MaxPoolNDDerivatives(BaseDerivatives):
    def __init__(self, N):
        self.N = N
        if self.N == 1:
            self.maxpool = max_pool1d
        elif self.N == 2:
            self.maxpool = max_pool2d
        elif self.N == 3:
            self.maxpool = max_pool3d
        else:
            raise ValueError(
                "{}-dimensional Maxpool. is not implemented.".format(self.N)
            )

    # TODO: Do not recompute but get from forward pass of module
    def get_pooling_idx(self, module):
        _, pool_idx = self.maxpool(
            module.input0,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True,
            ceil_mode=module.ceil_mode,
        )
        return pool_idx

    def hessian_is_zero(self):
        return True

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

        if self.N == 1:
            N, C, L_in = module.input0.size()
            _, _, L_out = module.output.size()
            in_pixels = L_in
            out_pixels = L_out
        elif self.N == 2:
            N, C, H_in, W_in = module.input0.size()
            _, _, H_out, W_out = module.output.size()
            in_pixels = H_in * W_in
            out_pixels = H_out * W_out
        elif self.N == 3:
            N, C, D_in, H_in, W_in = module.input0.size()
            _, _, D_out, H_out, W_out = module.output.size()
            in_pixels = D_in * H_in * W_in
            out_pixels = D_out * H_out * W_out

        in_features = C * in_pixels

        pool_idx = self.get_pooling_idx(module).view(N, C, out_pixels)

        def sample_ea_jac_t_mat_jac_prod(n, mat):
            jac_t_mat = sample_jac_t_mat_prod(n, mat)
            mat_t_jac = jac_t_mat.t()
            jac_t_mat_t_jac = sample_jac_t_mat_prod(n, mat_t_jac)
            return jac_t_mat_t_jac.t()

        def sample_jac_t_mat_prod(n, mat):
            num_cols = mat.size(1)
            idx = pool_idx[n, :, :].unsqueeze(-1).expand(-1, -1, num_cols)

            jac_t_mat = zeros(C, in_pixels, num_cols, device=device)
            mat = mat.reshape(C, out_pixels, num_cols)

            jac_t_mat.scatter_add_(1, idx, mat)

            return jac_t_mat.reshape(in_features, num_cols)

        result = zeros(in_features, in_features, device=device)

        for n in range(N):
            result += sample_ea_jac_t_mat_jac_prod(n, mat)

        return result / N

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = rearrange(mat, "v n c ... -> v n c (...)")
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        return self.reshape_like_output(jmp_as_pool, module)

    def __apply_jacobian_of(self, module, mat):
        V, N_axis = mat.shape[0], 3
        pool_idx = self.__pool_idx_for_jac(module, V)
        return mat.gather(N_axis, pool_idx)

    def __pool_idx_for_jac(self, module, V):
        """Manipulated pooling indices ready-to-use in jac(t)."""
        pool_idx = self.get_pooling_idx(module)
        pool_idx = rearrange(pool_idx, "n c ... -> n c (...)")

        V_axis = 0

        return pool_idx.unsqueeze(V_axis).expand(V, -1, -1, -1)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_pool = rearrange(mat, "v n c ... -> v n c (...)")
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        return self.reshape_like_input(jmp_as_pool, module)

    def __apply_jacobian_t_of(self, module, mat):
        V = mat.shape[0]
        result = self.__zero_for_jac_t(module, V, mat.device)
        pool_idx = self.__pool_idx_for_jac(module, V)

        N_axis = 3
        result.scatter_add_(N_axis, pool_idx, mat)
        return result

    def __zero_for_jac_t(self, module, V, device):
        if self.N == 1:
            N, C_out, _ = module.output.shape
            _, _, L_in = module.input0.size()

            shape = (V, N, C_out, L_in)

        elif self.N == 2:
            N, C_out, _, _ = module.output.shape
            _, _, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, H_in * W_in)

        elif self.N == 3:
            N, C_out, _, _, _ = module.output.shape
            _, _, D_in, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, D_in * H_in * W_in)

        return zeros(shape, device=device)

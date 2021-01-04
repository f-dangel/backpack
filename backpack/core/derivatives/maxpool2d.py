from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives
from torch import zeros


class MaxPool2DDerivatives(MaxPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=2)

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

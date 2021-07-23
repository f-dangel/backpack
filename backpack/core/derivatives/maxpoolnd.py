from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor, zeros
from torch.nn import MaxPool1d, MaxPool2d, MaxPool3d
from torch.nn.functional import max_pool1d, max_pool2d, max_pool3d

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.subsampling import subsample


class MaxPoolNDDerivatives(BaseDerivatives):
    def __init__(self, N: int):
        self.N = N
        self.maxpool = {
            1: max_pool1d,
            2: max_pool2d,
            3: max_pool3d,
        }[N]

    # TODO: Do not recompute but get from forward pass of module
    def get_pooling_idx(
        self,
        module: Union[MaxPool1d, MaxPool2d, MaxPool3d],
        subsampling: List[int] = None,
    ) -> Tensor:
        _, pool_idx = self.maxpool(
            subsample(module.input0, subsampling=subsampling),
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True,
            ceil_mode=module.ceil_mode,
        )
        return pool_idx

    def hessian_is_zero(self, module):
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

        N, C = module.input0.shape[:2]
        in_pixels = module.input0.shape[2:].numel()
        out_pixels = module.output.shape[2:].numel()
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

    def __pool_idx_for_jac(
        self,
        module: Union[MaxPool1d, MaxPool2d, MaxPool3d],
        V: int,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Manipulated pooling indices ready-to-use in jac(t)."""
        pool_idx = self.get_pooling_idx(module, subsampling=subsampling)
        pool_idx = rearrange(pool_idx, "n c ... -> n c (...)")

        return pool_idx.unsqueeze(0).expand(V, -1, -1, -1)

    def _jac_t_mat_prod(
        self,
        module: Union[MaxPool1d, MaxPool2d, MaxPool3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        mat_as_pool = rearrange(mat, "v n c ... -> v n c (...)")
        jmp_as_pool = self.__apply_jacobian_t_of(
            module, mat_as_pool, subsampling=subsampling
        )
        return self.reshape_like_input(jmp_as_pool, module, subsampling=subsampling)

    def __apply_jacobian_t_of(
        self,
        module: Union[MaxPool1d, MaxPool2d, MaxPool3d],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        V = mat.shape[0]
        result = self.__zero_for_jac_t(module, V, subsampling=subsampling)
        pool_idx = self.__pool_idx_for_jac(module, V, subsampling=subsampling)

        N_axis = 3
        result.scatter_add_(N_axis, pool_idx, mat)
        return result

    def __zero_for_jac_t(
        self,
        module: Union[MaxPool1d, MaxPool2d, MaxPool3d],
        V: int,
        subsampling: List[int] = None,
    ) -> Tensor:
        N, C_out = module.output.shape[:2]
        in_pixels = module.input0.shape[2:].numel()
        N = N if subsampling is None else len(subsampling)

        shape = (V, N, C_out, in_pixels)

        return zeros(shape, device=module.output.device, dtype=module.output.dtype)

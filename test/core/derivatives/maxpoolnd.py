from torch import zeros
from torch.nn.functional import max_pool1d, max_pool2d, max_pool3d

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.ein import eingroup


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
            raise ValueError("{}-dimensional Maxpool. is not implemented.".format(self.N))

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

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        if self.N == 1:
            mat_as_pool = mat
        elif self.N == 2:
            mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        elif self.N == 3:
            mat_as_pool = eingroup("v,n,c,d,h,w->v,n,c,dhw", mat)
        else:
            raise ValueError("{}-dimensional Maxpool. is not implemented.".format(self.N))
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        return self.reshape_like_output(jmp_as_pool, module)

    def __apply_jacobian_of(self, module, mat):
        V, N_axis = mat.shape[0], 3
        pool_idx = self.__pool_idx_for_jac(module, V)
        return mat.gather(N_axis, pool_idx)

    def __pool_idx_for_jac(self, module, V):
        """Manipulated pooling indices ready-to-use in jac(t)."""

        pool_idx = self.get_pooling_idx(module)
        V_axis = 0
        if self.N == 1:
            return (
                pool_idx
                .unsqueeze(V_axis)
                .expand(V, -1, -1, -1)
            )
        elif self.N == 2:
            return (
                eingroup("n,c,h,w->n,c,hw", pool_idx)
                .unsqueeze(V_axis)
                .expand(V, -1, -1, -1)
            )
        elif self.N == 3:
            return (
                eingroup("n,c,d,h,w->n,c,dhw", pool_idx)
                .unsqueeze(V_axis)
                .expand(V, -1, -1, -1)
            )
        else:
            raise ValueError("{}-dimensional Maxpool. is not implemented.".format(self.N))

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        if self.N == 1:
            mat_as_pool = mat
        elif self.N == 2:
            mat_as_pool = eingroup("v,n,c,h,w->v,n,c,hw", mat)
        elif self.N == 3:
            mat_as_pool = eingroup("v,n,c,d,h,w->v,n,c,dhw", mat)
        else:
            raise ValueError("{}-dimensional Maxpool. is not implemented.".format(self.N))
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
            return zeros(shape, device=device)

        elif self.N == 2:
            N, C_out, _, _ = module.output.shape
            _, _, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, H_in * W_in)
            return zeros(shape, device=device)

        elif self.N == 3:
            N, C_out, _, _, _ = module.output.shape
            _, _, D_in, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, D_in * H_in * W_in)
            return zeros(shape, device=device)
        else:
            raise ValueError("{}-dimensional Maxpool. is not implemented.".format(self.N))

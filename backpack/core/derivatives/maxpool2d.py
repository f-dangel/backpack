from torch import prod, zeros
from torch.nn import MaxPool2d
from torch.nn.functional import max_pool2d

from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from ...utils import conv as convUtils
from .basederivatives import BaseDerivatives


class MaxPool2DDerivatives(BaseDerivatives):
    def get_module(self):
        return MaxPool2d

    def get_pooling_idx(self, module):
        # TODO: Do not recompute but get from forward pass of module
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
        batch, channels, in_x, in_y = module.input0.size()
        in_features = channels * in_x * in_y
        _, _, out_x, out_y = module.output.size()
        out_features = channels * out_x * out_y

        pool_idx = self.get_pooling_idx(module).view(batch, channels, out_x * out_y)
        result = zeros(in_features, in_features, device=device)

        for b in range(batch):
            idx = pool_idx[b, :, :]
            temp = zeros(in_features, out_features, device=device)
            temp.scatter_add_(1, idx, mat)
            result.scatter_add_(0, idx.t(), temp)
        return result / batch

    def hessian_is_zero(self):
        return True

    # Jacobian-matrix product
    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        convUtils.check_sizes_input_jac(mat, module, new_convention=new_convention)
        mat_as_pool = self.__reshape_for_pooling_in(
            mat, module, new_convention=new_convention
        )
        jmp_as_pool = self.__apply_jacobian_of(
            module, mat_as_pool, new_convention=new_convention
        )
        return self.__reshape_for_matmul(
            jmp_as_pool, module, new_convention=new_convention
        )

    def __reshape_for_pooling_in(self, mat, module, new_convention=False):
        batch, channels, in_x, in_y = module.input0.size()

        if new_convention:
            num_classes = mat.size(0)
            shape = (num_classes, batch, channels, in_x * in_y)
        else:
            num_classes = mat.size(-1)
            shape = (batch, channels, in_x * in_y, num_classes)
        return mat.view(shape)

    def __reshape_for_matmul(self, mat, module, new_convention=False):
        if new_convention:
            num_columns = mat.size(0)
            shape = (num_columns,) + tuple(module.output_shape)
        else:
            batch = module.output_shape[0]
            out_features = prod(module.output_shape) / batch
            num_classes = mat.size(-1)
            shape = (batch, out_features, num_classes)
        return mat.view(shape)

    def __apply_jacobian_of(self, module, mat, new_convention=False):
        batch, channels, out_x, out_y = module.output_shape

        if new_convention:
            num_classes = mat.shape[0]
        else:
            num_classes = mat.shape[-1]

        pool_idx = self.get_pooling_idx(module)

        pool_idx = pool_idx.view(batch, channels, out_x * out_y)

        if new_convention:
            pool_idx = pool_idx.unsqueeze(0).expand(num_classes, -1, -1, -1)
            return mat.gather(3, pool_idx)
        else:
            pool_idx = pool_idx.unsqueeze(-1).expand(-1, -1, -1, num_classes)
            return mat.gather(2, pool_idx)

    # Transposed Jacobian-matrix product
    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True
        convUtils.check_sizes_input_jac_t(mat, module, new_convention=new_convention)
        mat_as_pool = self.__reshape_for_pooling_out(
            mat, module, new_convention=new_convention
        )
        jmp_as_pool = self.__apply_jacobian_t_of(
            module, mat_as_pool, new_convention=new_convention
        )
        return self.__reshape_for_matmul_t(
            jmp_as_pool, module, new_convention=new_convention
        )

    def __reshape_for_pooling_out(self, mat, module, new_convention=False):
        batch, channels, out_x, out_y = module.output_shape

        if new_convention:
            num_classes = mat.size(0)
            shape = (num_classes, batch, channels, out_x * out_y)
        else:
            num_classes = mat.size(-1)
            shape = (batch, channels, out_x * out_y, num_classes)
        return mat.view(shape)

    def __reshape_for_matmul_t(self, mat, module, new_convention=False):
        batch = module.output_shape[0]

        if new_convention:
            in_features_shape = module.input0_shape[1:]
            num_classes = mat.size(0)
            shape = (num_classes, batch) + tuple(in_features_shape)
        else:
            in_features = module.input0.numel() / batch
            num_classes = mat.size(-1)
            shape = (batch, in_features, num_classes)

        return mat.view(shape)

    def __apply_jacobian_t_of(self, module, mat, new_convention=False):
        batch, channels, out_x, out_y = module.output_shape
        _, _, in_x, in_y = module.input0.size()

        if new_convention:
            num_classes = mat.shape[0]
            shape = (num_classes, batch, channels, in_x * in_y)
        else:
            num_classes = mat.shape[-1]
            shape = (batch, channels, in_x * in_y, num_classes)

        result = zeros(shape, device=mat.device)

        pool_idx = self.get_pooling_idx(module)
        pool_idx = pool_idx.view(batch, channels, out_x * out_y)

        if new_convention:
            pool_idx = pool_idx.unsqueeze(0).expand(num_classes, -1, -1, -1)
            result.scatter_add_(3, pool_idx, mat)
        else:
            pool_idx = pool_idx.unsqueeze(-1).expand(-1, -1, -1, num_classes)
            result.scatter_add_(2, pool_idx, mat)

        return result

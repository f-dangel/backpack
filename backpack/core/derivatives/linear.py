from torch.nn import Linear

from backpack.core.derivatives.utils import (
    weight_jac_t_mat_prod_accept_vectors,
    weight_jac_mat_prod_accept_vectors,
    bias_jac_t_mat_prod_accept_vectors,
    bias_jac_mat_prod_accept_vectors,
)

from backpack.utils.einsum import einsum
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class LinearDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the Linear layer.

    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def get_module(self):
        return Linear

    def hessian_is_zero(self):
        return True

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """Apply transposed Jacobian of the output w.r.t. the input."""
        d_input = module.weight.data
        return einsum("oi,vno->vni", (d_input, mat))

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the input."""
        d_input = module.weight.data
        return einsum("oi,vni->vno", (d_input, mat))

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        jac = module.weight.data
        return einsum("ik,ij,jl->kl", (jac, mat, jac))

    @weight_jac_mat_prod_accept_vectors
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the weight."""
        d_weight = module.input0
        return einsum("ni,voi->vno", (d_weight, mat))

    @weight_jac_t_mat_prod_accept_vectors
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. the weight."""
        d_weight = module.input0
        contract = "vno,ni->voi" if sum_batch else "vno,ni->vnoi"
        return einsum(contract, (mat, d_weight))

    @bias_jac_mat_prod_accept_vectors
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the bias."""
        N = self.get_batch(module)
        return mat.unsqueeze(1).expand(-1, N, -1)

    @bias_jac_t_mat_prod_accept_vectors
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. the bias."""
        if sum_batch:
            N_axis = 1
            return mat.sum(N_axis)
        else:
            return mat

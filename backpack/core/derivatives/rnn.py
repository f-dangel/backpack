from torch import rand

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class RNNDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the torch.nn.RNN layer.

    Index conventions:
    ------------------
    * t: Sequence dimension
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def _bias_ih_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. bias_ih_l0."""
        V = mat.shape[0]
        N = mat.shape[2]
        if sum_batch:
            return rand(V, *module.bias_ih_l0.shape)
        else:
            return rand(V, N, *module.bias_ih_l0.shape)

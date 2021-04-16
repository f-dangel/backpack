from torch import diag_embed, einsum, zeros

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class RNNDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the torch.nn.RNN layer.

    Index conventions:
    ------------------
    * t: Sequence dimension
    * v: Free dimension
    * n: Batch dimension
    * h: Output dimension
    * i: Input dimension
    """

    def _bias_ih_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. bias_ih_l0."""
        # V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = mat.shape[3]
        output = module.output
        jac = zeros(T, N, H, H)
        for t in range(T):
            jac[t, ...] = diag_embed(1 - output[t, ...] ** 2, dim1=1, dim2=2)
            if t > 0:
                jac[t, ...] += einsum(
                    "nh, hl, nkl -> nkh",
                    1 - output[t, ...] ** 2,
                    module.weight_hh_l0,
                    jac[t - 1, ...],
                )
        if sum_batch:
            eq = "tnhk, vtnk -> vh"
        else:
            eq = "tnhk, vtnk -> vnh"
        grad = einsum(eq, jac, mat)
        return grad

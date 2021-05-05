"""Partial derivatives for the torch.nn.RNN layer."""
from torch import diag_embed, einsum, eye, zeros

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

    def _check_parameters(self, module):
        """Check the parameters of module.

        Args:
            module (torch.nn.RNN): module which to check

        Raises:
            ValueError: If any parameter of module does not match expectation
        """
        if module.num_layers > 1:
            raise ValueError("only num_layers = 1 is supported")
        if not module.nonlinearity == "tanh":
            raise ValueError("only nonlinearity = tanh is supported")
        if module.bias is not True:
            raise ValueError("only bias = True is supported")
        if module.batch_first is not False:
            raise ValueError("only batch_first = False is supported")
        if not module.dropout == 0:
            raise ValueError("only dropout = 0 is supported")
        if module.bidirectional is not False:
            raise ValueError("only bidirectional = False is supported")

    def _bias_ih_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. bias_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            torch.nn.Tensor: product
        """
        self._check_parameters(module)
        # V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = mat.shape[3]
        output = module.output
        jac = zeros(T, N, H, H, device=mat.device)
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

    def _bias_hh_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. bias_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            torch.nn.Tensor: product
        """
        self._check_parameters(module)
        # identical to bias_ih_l0
        return self._bias_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, mat, sum_batch=sum_batch
        )

    def _weight_ih_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. weight_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            torch.nn.Tensor: product
        """
        self._check_parameters(module)
        # V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = mat.shape[3]
        output = module.output
        input0 = module.input0
        jac = zeros(T, N, H, H, module.input_size, device=mat.device)
        for t in range(T):
            jac[t, ...] = einsum(
                "nk, kh, nj -> nkhj",
                1 - output[t, ...] ** 2,
                eye(H, device=mat.device),
                input0[t],
            )
            if t > 0:
                jac[t, ...] += einsum(
                    "nh, hl, nklj -> nkhj",
                    1 - output[t, ...] ** 2,
                    module.weight_hh_l0,
                    jac[t - 1, ...],
                )
        if sum_batch:
            eq = "tnhki, vtnk -> vhi"
        else:
            eq = "tnhki, vtnk -> vnhi"
        grad = einsum(eq, jac, mat)
        return grad

    def _weight_hh_l0_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. weight_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            torch.nn.Tensor: product
        """
        self._check_parameters(module)
        # V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = mat.shape[3]
        output = module.output
        jac = zeros(T, N, H, H, H, device=mat.device)
        for t in range(T):
            if t > 0:
                jac[t, ...] = einsum(
                    "nk, kh, nj -> nkhj",
                    1 - output[t, ...] ** 2,
                    eye(H, device=mat.device),
                    output[t - 1],
                )
                jac[t, ...] += einsum(
                    "nh, hl, nklj -> nkhj",
                    1 - output[t, ...] ** 2,
                    module.weight_hh_l0,
                    jac[t - 1, ...],
                )
        if sum_batch:
            eq = "tnhkj, vtnk -> vhj"
        else:
            eq = "tnhkj, vtnk -> vnhj"
        grad = einsum(eq, jac, mat)
        return grad

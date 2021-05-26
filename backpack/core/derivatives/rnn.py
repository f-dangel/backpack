"""Partial derivatives for the torch.nn.RNN layer."""
from typing import Tuple

import torch
from torch import Tensor, cat, einsum, zeros
from torch.nn import RNN

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

    @staticmethod
    def _check_parameters(module):
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

    @staticmethod
    def _a_jac_t_mat_prod(
        output: Tensor,
        weight_hh_l0: Tensor,
        mat: Tensor,
    ) -> Tensor:
        """Calculates jacobian vector product wrt a.

        Args:
            output: the values of the hidden layer
            weight_hh_l0: weight matrix hidden-to-hidden
            mat: matrix to multiply

        Returns:
            jacobian vector product wrt a
        """
        V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = mat.shape[3]
        a_jac_t_mat_prod = zeros(V, T, N, H, device=mat.device)
        for t in range(T)[::-1]:
            if t == (T - 1):
                a_jac_t_mat_prod[:, t, ...] = einsum(
                    "vnh, nh -> vnh",
                    mat[:, t, ...],
                    1 - output[t, ...] ** 2,
                )
            else:
                a_jac_t_mat_prod[:, t, ...] = einsum(
                    "vnh, nh -> vnh",
                    mat[:, t, ...] + a_jac_t_mat_prod[:, t + 1, ...] @ weight_hh_l0,
                    1 - output[t, ...] ** 2,
                )
        return a_jac_t_mat_prod

    def _jac_t_mat_prod(
        self, module: RNN, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        eq = "vtnh, hk -> vtnk"
        return torch.einsum(
            eq,
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            module.weight_ih_l0,
        )

    def _jac_mat_prod(
        self, module: RNN, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        V = mat.shape[0]
        N = mat.shape[2]
        T = mat.shape[1]
        H = module.hidden_size
        _jac_mat_prod = torch.zeros(V, T, N, H, device=mat.device)
        for t in range(T):
            if t == 0:
                _jac_mat_prod[:, t, ...] = einsum(
                    "nh, hi, vni -> vnh",
                    1 - module.output[t, ...] ** 2,
                    module.weight_ih_l0,
                    mat[:, t, ...],
                )
            else:
                _jac_mat_prod[:, t, ...] = einsum(
                    "nh, vnh -> vnh",
                    1 - module.output[t, ...] ** 2,
                    einsum("hi, vni -> vnh", module.weight_ih_l0, mat[:, t, ...])
                    + einsum(
                        "hk, vnk -> vnh",
                        module.weight_hh_l0,
                        _jac_mat_prod[:, t - 1, ...],
                    ),
                )
        return _jac_mat_prod

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
        if sum_batch:
            dim = [1, 2]
        else:
            dim = 1
        return self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat).sum(
            dim=dim
        )

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
        if sum_batch:
            eq = "vtnh, tnj -> vhj"
        else:
            eq = "vtnh, tnj -> vnhj"
        return einsum(
            eq,
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            module.input0,
        )

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
        # T = mat.shape[1]
        H = mat.shape[3]
        if sum_batch:
            eq = "vtnh, tnk -> vhk"
        else:
            eq = "vtnh, tnk -> vnhk"
        return einsum(
            eq,
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            cat([zeros(1, N, H, device=mat.device), module.output[0:-1]], dim=0),
        )
"""Partial derivatives for the torch.nn.RNN layer."""
from typing import List, Tuple

import torch
from torch import Tensor, cat, einsum, zeros
from torch.nn import RNN

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class RNNDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the torch.nn.RNN layer.

    a_t = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
    h_t = tanh(a_t)

    Index conventions:
    ------------------
    * t: Sequence dimension
    * v: Free dimension
    * n: Batch dimension
    * h: Output dimension
    * i: Input dimension
    """

    @staticmethod
    def _check_parameters(module: RNN) -> None:
        """Check the parameters of module.

        Args:
            module: module which to check

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
        V: int = mat.shape[0]
        N: int = mat.shape[2]
        T: int = mat.shape[1]
        H: int = mat.shape[3]
        a_jac_t_mat_prod: Tensor = zeros(V, T, N, H, device=mat.device)
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
                    mat[:, t, ...]
                    + einsum(
                        "vng, gh -> vnh",
                        a_jac_t_mat_prod[:, t + 1, ...],
                        weight_hh_l0,
                    ),
                    1 - output[t, ...] ** 2,
                )
        return a_jac_t_mat_prod

    def _jac_t_mat_prod(
        self, module: RNN, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        return torch.einsum(
            "vtnh, hk -> vtnk",
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            module.weight_ih_l0,
        )

    def _jac_mat_prod(
        self, module: RNN, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        V: int = mat.shape[0]
        N: int = mat.shape[2]
        T: int = mat.shape[1]
        H: int = module.hidden_size
        _jac_mat_prod: Tensor = torch.zeros(V, T, N, H, device=mat.device)
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

    def _bias_ih_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. bias_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            product
        """
        self._check_parameters(module)
        if sum_batch:
            dim: List[int] = [1, 2]
        else:
            dim: int = 1
        return self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat).sum(
            dim=dim
        )

    def _bias_hh_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. bias_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            product
        """
        # identical to bias_ih_l0
        return self._bias_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, mat, sum_batch=sum_batch
        )

    def _weight_ih_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. weight_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            product
        """
        self._check_parameters(module)
        return einsum(
            "vtnh, tnj ->" + ("vhj" if sum_batch else "vnhj"),
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            module.input0,
        )

    def _weight_hh_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. weight_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.

        Returns:
            product
        """
        self._check_parameters(module)
        N: int = mat.shape[2]
        H: int = mat.shape[3]
        return einsum(
            "vtnh, tnk -> " + ("vhk" if sum_batch else "vnhk"),
            self._a_jac_t_mat_prod(module.output, module.weight_hh_l0, mat),
            cat([zeros(1, N, H, device=mat.device), module.output[0:-1]], dim=0),
        )

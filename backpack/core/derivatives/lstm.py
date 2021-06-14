"""Partial derivatives for nn.LSTM."""
from typing import Tuple

from torch import Tensor, allclose, cat, einsum, sigmoid, tanh, zeros
from torch.nn import LSTM

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class LSTMDerivatives(BaseParameterDerivatives):
    """Partial derivatives for nn.LSTM layer.

    Index conventions:
    ------------------
    * t: Sequence dimension
    * v: Free dimension
    * n: Batch dimension
    * h: Output dimension
    * i: Input dimension
    """

    @staticmethod
    def _check_parameters(module: LSTM) -> None:
        """Check the parameters of module.

        Args:
            module: module which to check

        Raises:
            ValueError: If any parameter of module does not match expectation
        """
        if module.num_layers > 1:
            raise ValueError("only num_layers = 1 is supported")
        if module.bias is not True:
            raise ValueError("only bias = True is supported")
        if module.batch_first is not False:
            raise ValueError("only batch_first = False is supported")
        if not module.dropout == 0:
            raise ValueError("only dropout = 0 is supported")
        if module.bidirectional is not False:
            raise ValueError("only bidirectional = False is supported")
        if not module.proj_size == 0:
            raise ValueError("only proj_size = 0 is supported")

    @staticmethod
    def _forward_pass(
        module: LSTM, mat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        T: int = mat.shape[1]
        N: int = mat.shape[2]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H
        # forward pass and save i, f, g, o, c, c_tanh-> ifgo, c, c_tanh
        ifgo: Tensor = zeros(T, N, 4 * H, device=mat.device)
        c: Tensor = zeros(T, N, H, device=mat.device)
        c_tanh: Tensor = zeros(T, N, H, device=mat.device)
        h: Tensor = zeros(T, N, H, device=mat.device)
        for t in range(T):
            if t == 0:
                ifgo[t] = (
                    einsum("hi, ni -> nh", module.weight_ih_l0, module.input0[t])
                    + module.bias_ih_l0
                    + module.bias_hh_l0
                )
            else:
                ifgo[t] = (
                    einsum("hi, ni -> nh", module.weight_ih_l0, module.input0[t])
                    + module.bias_ih_l0
                    + einsum("hg, ng -> nh", module.weight_hh_l0, module.output[t - 1])
                    + module.bias_hh_l0
                )
            ifgo[t, :, H0:H1] = sigmoid(ifgo[t, :, H0:H1])
            ifgo[t, :, H1:H2] = sigmoid(ifgo[t, :, H1:H2])
            ifgo[t, :, H2:H3] = tanh(ifgo[t, :, H2:H3])
            ifgo[t, :, H3:H4] = sigmoid(ifgo[t, :, H3:H4])
            if t == 0:
                c[t] = ifgo[t, :, :H] * ifgo[t, :, H2:H3]
            else:
                c[t] = (ifgo[t, :, :H] * ifgo[t, :, H2:H3]) + (
                    ifgo[t, :, H:H2] * c[t - 1]
                )
            c_tanh[t] = tanh(c[t])
            h[t] = ifgo[t, :, H3:] * c_tanh[t]

        # h is the same as previous forward pass
        assert allclose(h, module.output, atol=1e-5)
        return ifgo, c, c_tanh, h

    @classmethod
    def _ifgo_jac_t_mat_prod(cls, module: LSTM, mat: Tensor) -> Tensor:
        V: int = mat.shape[0]
        T: int = mat.shape[1]
        N: int = mat.shape[2]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H

        ifgo, c, c_tanh, _ = cls._forward_pass(module, mat)

        # backward pass
        H_prod: Tensor = zeros(V, T, N, H, device=mat.device)
        C_prod: Tensor = zeros(V, T, N, H, device=mat.device)
        IFGO_prod: Tensor = zeros(V, T, N, 4 * H, device=mat.device)
        for t in range(T)[::-1]:
            # jac_t_mat_prod until node h
            if t == T - 1:
                H_prod[:, t] = mat[:, t]
            else:
                H_prod[:, t] = mat[:, t] + einsum(
                    "vnh, hg -> vng",
                    IFGO_prod[:, t + 1],
                    module.weight_hh_l0,
                )

            # C_prod = jac_t_mat_prod until node c
            if t == T - 1:
                C_prod[:, t] = einsum(
                    "vnh, nh, nh -> vnh",
                    H_prod[:, t],
                    ifgo[t, :, H3:H4],
                    (1 - c_tanh[t] ** 2),
                )
            else:
                C_prod[:, t] = einsum(
                    "vnh, nh, nh -> vnh",
                    H_prod[:, t],
                    ifgo[t, :, H3:H4],
                    (1 - c_tanh[t] ** 2),
                ) + einsum(
                    "vnh, nh -> vnh",
                    C_prod[:, t + 1],
                    ifgo[t + 1, :, H1:H2],
                )

            IFGO_prod[:, t, :, H3:] = einsum(
                "vnh, nh, nh -> vnh",
                H_prod[:, t],
                c_tanh[t],
                ifgo[t, :, H3:H4] * (1 - ifgo[t, :, H3:H4]),
            )
            IFGO_prod[:, t, :, :H] = einsum(
                "vnh, nh, nh -> vnh",
                C_prod[:, t],
                ifgo[t, :, H2:H3],
                ifgo[t, :, H0:H1] * (1 - ifgo[t, :, H0:H1]),
            )
            if t >= 1:
                IFGO_prod[:, t, :, H1:H2] = einsum(
                    "vnh, nh, nh -> vnh",
                    C_prod[:, t],
                    c[t - 1],
                    ifgo[t, :, H1:H2] * (1 - ifgo[t, :, H1:H2]),
                )
            IFGO_prod[:, t, :, H2:H3] = einsum(
                "vnh, nh, nh -> vnh",
                C_prod[:, t],
                ifgo[t, :, H0:H1],
                1 - ifgo[t, :, H2:H3] ** 2,
            )
        return IFGO_prod

    def _jac_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        V: int = mat.shape[0]
        T: int = mat.shape[1]
        N: int = mat.shape[2]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H

        ifgo, c, c_tanh, h = self._forward_pass(module, mat)
        H_prod: Tensor = zeros(V, T, N, H, device=mat.device)
        C_prod: Tensor = zeros(V, T, N, H, device=mat.device)
        C_tanh_prod_t: Tensor = zeros(V, N, H, device=mat.device)
        IFGO_prod_t: Tensor = zeros(V, N, 4 * H, device=mat.device)
        for t in range(T):
            # product until nodes ifgo
            if t == 0:
                IFGO_prod_t[:] = einsum(
                    "hi, vni -> vnh",
                    module.weight_ih_l0,
                    mat[:, t],
                )
            else:
                IFGO_prod_t[:] = einsum(
                    "hi, vni -> vnh",
                    module.weight_ih_l0,
                    mat[:, t],
                ) + einsum(
                    "hg, vng -> vnh",
                    module.weight_hh_l0,
                    H_prod[:, t - 1],
                )
            IFGO_prod_t[:, :, H0:H2] = einsum(
                "vnh, nh -> vnh",
                IFGO_prod_t[:, :, H0:H2],
                ifgo[t, :, H0:H2] * (1 - ifgo[t, :, H0:H2]),
            )
            IFGO_prod_t[:, :, H3:H4] = einsum(
                "vnh, nh -> vnh",
                IFGO_prod_t[:, :, H3:H4],
                ifgo[t, :, H3:H4] * (1 - ifgo[t, :, H3:H4]),
            )
            IFGO_prod_t[:, :, H2:H3] = einsum(
                "vnh, nh -> vnh",
                IFGO_prod_t[:, :, H2:H3],
                1 - ifgo[t, :, H2:H3] ** 2,
            )

            # product until node c
            C_prod[:, t] = (
                einsum(
                    "vnh, nh -> vnh",
                    IFGO_prod_t[:, :, H0:H1],
                    ifgo[t, :, H2:H3],
                )
                + einsum("vnh, nh -> vnh", IFGO_prod_t[:, :, H2:H3], ifgo[t, :, H0:H1])
            )
            if t >= 1:
                C_prod[:, t] += einsum(
                    "vnh, nh -> vnh",
                    C_prod[:, t - 1],
                    ifgo[t, :, H1:H2],
                ) + einsum(
                    "vnh, nh -> vnh",
                    IFGO_prod_t[:, :, H1:H2],
                    c[t - 1],
                )

            # product until node c_tanh
            C_tanh_prod_t[:] = einsum(
                "vnh, nh -> vnh",
                C_prod[:, t],
                1 - c_tanh[t] ** 2,
            )

            # product until node h
            H_prod[:, t] = einsum(
                "vnh, nh -> vnh",
                IFGO_prod_t[:, :, H3:H4],
                c_tanh[t],
            ) + einsum(
                "vnh, nh -> vnh",
                C_tanh_prod_t,
                ifgo[t, :, H3:H4],
            )

        return H_prod

    def _jac_t_mat_prod(
        self, module: LSTM, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:

        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(module, mat)

        X_prod: Tensor = einsum(
            "vtnh, hi -> vtni",
            IFGO_prod,
            module.weight_ih_l0,
        )
        return X_prod

    def _bias_ih_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(module, mat)

        if sum_batch:
            eq = "vtnh -> vh"
        else:
            eq = "vtnh -> vnh"
        return einsum(eq, IFGO_prod)

    def _bias_hh_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(module, mat)

        if sum_batch:
            eq = "vtnh -> vh"
        else:
            eq = "vtnh -> vnh"
        return einsum(eq, IFGO_prod)

    def _weight_ih_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(module, mat)

        if sum_batch:
            eq = "vtnh, tni -> vhi"
        else:
            eq = "vtnh, tni -> vnhi"
        return einsum(eq, IFGO_prod, module.input0)

    def _weight_hh_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        self._check_parameters(module)

        N: int = mat.shape[2]
        H: int = module.hidden_size

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(module, mat)

        if sum_batch:
            eq = "vtnh, tng -> vhg"
        else:
            eq = "vtnh, tng -> vnhg"
        return einsum(
            eq,
            IFGO_prod,
            cat([zeros(1, N, H, device=mat.device), module.output[0:-1]], dim=0),
        )

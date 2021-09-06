"""Partial derivatives for nn.LSTM."""
from typing import List, Tuple

from torch import Tensor, cat, einsum, sigmoid, tanh, zeros
from torch.nn import LSTM

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import TORCH_VERSION_AT_LEAST_1_8_0
from backpack.utils.subsampling import subsample


class LSTMDerivatives(BaseParameterDerivatives):
    """Partial derivatives for nn.LSTM layer.

    Index conventions:
    ------------------
    * t: Sequence dimension
    * v: Free dimension
    * n: Batch dimension
    * h: Output dimension
    * i: Input dimension

    LSTM forward pass (definition of variables):
    see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    ifgo_tilde[t] = W_ih x[t] + b_ii + W_hh h[t-1] + b_hh
    ifgo[t] = sigma(ifgo_tilde[t]) for i, f, o
    ifgo[t] = tanh(ifgo_tilde[t]) for g
    c[t] = f[t] c[t-1] + i[t] g[t]
    h[t] = o[t] tanh(c[t])

    Note:
        For ``batch_first=True``, most of the internal tensors (e.g. those from
        the manual forward pass) are kept with time axis first.
    """

    @staticmethod
    def _check_parameters(module: LSTM) -> None:
        """Check the parameters of module.

        Args:
            module: module which to check

        Raises:
            NotImplementedError: If any parameter of module does not match expectation
        """
        if module.num_layers != 1:
            raise NotImplementedError("only num_layers = 1 is supported")
        if module.bias is not True:
            raise NotImplementedError("only bias = True is supported")
        if module.dropout != 0:
            raise NotImplementedError("only dropout = 0 is supported")
        if module.bidirectional is not False:
            raise NotImplementedError("only bidirectional = False is supported")
        if TORCH_VERSION_AT_LEAST_1_8_0:
            if module.proj_size != 0:
                raise NotImplementedError("only proj_size = 0 is supported")

    @staticmethod
    def _forward_pass(
        module: LSTM, mat: Tensor, subsampling: List[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """This performs an additional forward pass and returns the hidden variables.

        This is important because the PyTorch implementation does not grant access to
        some of the hidden variables. Those are computed and returned.

        See also forward pass in class docstring.

        Args:
            module: module
            mat: matrix, used to extract device and shapes.
            subsampling: Indices of active samples. Defaults to ``None`` (all samples).

        Returns:
            ifgo, c, c_tanh (all in format ``[T, N, ...]``)
        """
        free_axis = 1
        N_axis, T_axis = LSTMDerivatives.get_batch_and_time_axes(module)
        T: int = mat.shape[T_axis + free_axis]
        N: int = mat.shape[N_axis + free_axis]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H
        # forward pass and save i, f, g, o, c, c_tanh-> ifgo, c, c_tanh
        ifgo: Tensor = zeros(T, N, 4 * H, device=mat.device, dtype=mat.dtype)
        c: Tensor = zeros(T, N, H, device=mat.device, dtype=mat.dtype)
        c_tanh: Tensor = zeros(T, N, H, device=mat.device, dtype=mat.dtype)

        input0 = subsample(module.input0, dim=N_axis, subsampling=subsampling)
        output = subsample(module.output, dim=N_axis, subsampling=subsampling)

        # use [T, N, ...] format
        if module.batch_first:
            input0 = input0.transpose(N_axis, T_axis)
            output = output.transpose(N_axis, T_axis)

        for t in range(T):
            ifgo[t] = (
                einsum("hi,ni->nh", module.weight_ih_l0, input0[t])
                + module.bias_ih_l0
                + module.bias_hh_l0
            )
            if t != 0:
                ifgo[t] += einsum("hg,ng->nh", module.weight_hh_l0, output[t - 1])
            ifgo[t, :, H0:H1] = sigmoid(ifgo[t, :, H0:H1])
            ifgo[t, :, H1:H2] = sigmoid(ifgo[t, :, H1:H2])
            ifgo[t, :, H2:H3] = tanh(ifgo[t, :, H2:H3])
            ifgo[t, :, H3:H4] = sigmoid(ifgo[t, :, H3:H4])
            c[t] = ifgo[t, :, H0:H1] * ifgo[t, :, H2:H3]
            if t != 0:
                c[t] += ifgo[t, :, H1:H2] * c[t - 1]
            c_tanh[t] = tanh(c[t])

        return ifgo, c, c_tanh

    @classmethod
    def _ifgo_jac_t_mat_prod(
        cls, module: LSTM, mat: Tensor, subsampling: List[int] = None
    ) -> Tensor:
        free_axis = 1
        N_axis, T_axis = cls.get_batch_and_time_axes(module)
        V: int = mat.shape[0]
        T: int = mat.shape[T_axis + free_axis]
        N: int = mat.shape[N_axis + free_axis]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H

        ifgo, c, c_tanh = cls._forward_pass(module, mat, subsampling=subsampling)

        # backward pass
        H_prod_t: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        C_prod_t: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        C_prod_old: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        IFGO_prod: Tensor = zeros(V, T, N, 4 * H, device=mat.device, dtype=mat.dtype)
        for t in reversed(range(T)):
            # jac_t_mat_prod until node h
            H_prod_t[:] = mat[(slice(None),) * (T_axis + 1) + (t,)]
            if t != (T - 1):
                H_prod_t += einsum(
                    "vnh,hg->vng", IFGO_prod[:, t + 1], module.weight_hh_l0
                )

            # C_prod_t = jac_t_mat_prod until node c
            if t != (T - 1):
                C_prod_old[:] = C_prod_t
            C_prod_t[:] = einsum(
                "vnh,nh->vnh", H_prod_t, ifgo[t, :, H3:H4] * (1 - c_tanh[t] ** 2)
            )
            if t != (T - 1):
                C_prod_t += einsum("vnh,nh->vnh", C_prod_old, ifgo[t + 1, :, H1:H2])

            IFGO_prod[:, t, :, H3:H4] = einsum(
                "vnh,nh->vnh",
                H_prod_t,
                c_tanh[t] * (ifgo[t, :, H3:H4] * (1 - ifgo[t, :, H3:H4])),
            )
            IFGO_prod[:, t, :, H0:H1] = einsum(
                "vnh,nh->vnh",
                C_prod_t,
                ifgo[t, :, H2:H3] * (ifgo[t, :, H0:H1] * (1 - ifgo[t, :, H0:H1])),
            )
            if t >= 1:
                IFGO_prod[:, t, :, H1:H2] = einsum(
                    "vnh,nh->vnh",
                    C_prod_t,
                    c[t - 1] * (ifgo[t, :, H1:H2] * (1 - ifgo[t, :, H1:H2])),
                )
            IFGO_prod[:, t, :, H2:H3] = einsum(
                "vnh,nh->vnh",
                C_prod_t,
                ifgo[t, :, H0:H1] * (1 - ifgo[t, :, H2:H3] ** 2),
            )
        return IFGO_prod

    def hessian_is_zero(self, module: LSTM) -> bool:  # noqa: D102
        return False

    def _jac_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        free_axis = 1
        N_axis, T_axis = self.get_batch_and_time_axes(module)
        V: int = mat.shape[0]
        T: int = mat.shape[T_axis + free_axis]
        N: int = mat.shape[N_axis + free_axis]
        H: int = module.hidden_size
        H0: int = 0 * H
        H1: int = 1 * H
        H2: int = 2 * H
        H3: int = 3 * H
        H4: int = 4 * H

        ifgo, c, c_tanh = self._forward_pass(module, mat)
        H_prod: Tensor = zeros(V, T, N, H, device=mat.device, dtype=mat.dtype)
        C_prod_t: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        C_prod_old: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        C_tanh_prod_t: Tensor = zeros(V, N, H, device=mat.device, dtype=mat.dtype)
        IFGO_prod_t: Tensor = zeros(V, N, 4 * H, device=mat.device, dtype=mat.dtype)
        for t in range(T):
            # product until nodes ifgo
            IFGO_prod_t[:] = einsum(
                "hi,vni->vnh",
                module.weight_ih_l0,
                mat[(slice(None),) * (T_axis + free_axis) + (t,)],
            )
            if t != 0:
                IFGO_prod_t[:] += einsum(
                    "hg,vng->vnh", module.weight_hh_l0, H_prod[:, t - 1]
                )
            IFGO_prod_t[:, :, H0:H2] = einsum(
                "vnh,nh->vnh",
                IFGO_prod_t[:, :, H0:H2],
                ifgo[t, :, H0:H2] * (1 - ifgo[t, :, H0:H2]),
            )
            IFGO_prod_t[:, :, H3:H4] = einsum(
                "vnh,nh->vnh",
                IFGO_prod_t[:, :, H3:H4],
                ifgo[t, :, H3:H4] * (1 - ifgo[t, :, H3:H4]),
            )
            IFGO_prod_t[:, :, H2:H3] = einsum(
                "vnh,nh->vnh",
                IFGO_prod_t[:, :, H2:H3],
                1 - ifgo[t, :, H2:H3] ** 2,
            )

            # product until node c
            if t >= 1:
                C_prod_old[:] = C_prod_t
            C_prod_t[:] = einsum(
                "vnh,nh->vnh", IFGO_prod_t[:, :, H0:H1], ifgo[t, :, H2:H3]
            ) + einsum("vnh,nh->vnh", IFGO_prod_t[:, :, H2:H3], ifgo[t, :, H0:H1])
            if t >= 1:
                C_prod_t += einsum(
                    "vnh,nh->vnh", C_prod_old, ifgo[t, :, H1:H2]
                ) + einsum("vnh,nh->vnh", IFGO_prod_t[:, :, H1:H2], c[t - 1])

            # product until node c_tanh
            C_tanh_prod_t[:] = einsum("vnh,nh->vnh", C_prod_t, 1 - c_tanh[t] ** 2)

            # product until node h
            H_prod[:, t] = einsum(
                "vnh,nh->vnh", IFGO_prod_t[:, :, H3:H4], c_tanh[t]
            ) + einsum("vnh,nh->vnh", C_tanh_prod_t, ifgo[t, :, H3:H4])

        if module.batch_first:
            H_prod = H_prod.transpose(T_axis + free_axis, N_axis + free_axis)

        return H_prod

    def _jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(
            module, mat, subsampling=subsampling
        )

        N_axis, _ = self.get_batch_and_time_axes(module)
        batch_time_str = "nt" if N_axis == 0 else "tn"

        X_prod: Tensor = einsum(
            f"vtnh,hi->v{batch_time_str}i", IFGO_prod, module.weight_ih_l0
        )
        return X_prod

    def _bias_ih_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(
            module, mat, subsampling=subsampling
        )

        return einsum(f"vtnh->v{'' if sum_batch else 'n'}h", IFGO_prod)

    def _bias_hh_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        return self._bias_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, mat, sum_batch=sum_batch, subsampling=subsampling
        )

    def _weight_ih_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(
            module, mat, subsampling=subsampling
        )

        N_axis, _ = self.get_batch_and_time_axes(module)
        batch_time_str = "nt" if N_axis == 0 else "tn"

        return einsum(
            f"vtnh,{batch_time_str}i->v{'' if sum_batch else 'n'}hi",
            IFGO_prod,
            subsample(module.input0, dim=N_axis, subsampling=subsampling),
        )

    def _weight_hh_l0_jac_t_mat_prod(
        self,
        module: LSTM,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)

        free_axis = 1
        N_axis, T_axis = self.get_batch_and_time_axes(module)

        N: int = mat.shape[N_axis + free_axis]
        H: int = module.hidden_size

        IFGO_prod: Tensor = self._ifgo_jac_t_mat_prod(
            module, mat, subsampling=subsampling
        )

        subsampled_output = subsample(
            module.output, dim=N_axis, subsampling=subsampling
        )
        if N_axis == 0:
            subsampled_output = subsampled_output.transpose(N_axis, T_axis)

        return einsum(
            f"vtnh,tng->v{'' if sum_batch else 'n'}hg",
            IFGO_prod,
            cat(
                [
                    zeros(1, N, H, device=mat.device, dtype=mat.dtype),
                    subsampled_output[0:-1],
                ],
                dim=0,
            ),
        )

    @staticmethod
    def get_batch_and_time_axes(module: LSTM) -> Tuple[int, int]:
        """Return axes interpreted by the module as batch and time axes of the input.

        Args:
            module: LSTM module.

        Returns:
            Batch axis and time axis.
        """
        return (0, 1) if module.batch_first else (1, 0)

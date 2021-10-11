"""Partial derivatives for the torch.nn.RNN layer."""
from typing import List, Tuple

from torch import Tensor, cat, einsum, zeros
from torch.nn import RNN

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class RNNDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the torch.nn.RNN layer.

    a_t = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
    h_t = tanh(a_t)

    We assume that it is always batch axis first.

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
            NotImplementedError: If any parameter of module does not match expectation
        """
        if not module.batch_first:
            raise NotImplementedError("Batch axis must be first.")
        if module.num_layers > 1:
            raise NotImplementedError("only num_layers = 1 is supported")
        if not module.nonlinearity == "tanh":
            raise NotImplementedError("only nonlinearity = tanh is supported")
        if module.bias is not True:
            raise NotImplementedError("only bias = True is supported")
        if not module.dropout == 0:
            raise NotImplementedError("only dropout = 0 is supported")
        if module.bidirectional is not False:
            raise NotImplementedError("only bidirectional = False is supported")

    def hessian_is_zero(self, module: RNN) -> bool:  # noqa: D102
        return False

    @classmethod
    def _a_jac_t_mat_prod(
        cls,
        module: RNN,
        weight_hh_l0: Tensor,
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Calculates jacobian vector product wrt a.

        Args:
            module: RNN module
            weight_hh_l0: weight matrix hidden-to-hidden
            mat: matrix to multiply
            subsampling: subsampling

        Returns:
            jacobian vector product wrt a
        """
        V, N, T, H = mat.shape
        output = subsample(module.output, dim=0, subsampling=subsampling)
        a_jac_t_mat_prod: Tensor = zeros(V, N, T, H, device=mat.device, dtype=mat.dtype)
        for t in reversed(range(T)):
            if t == (T - 1):
                a_jac_t_mat_prod[:, :, t] = einsum(
                    "vnh,nh->vnh", mat[:, :, t], 1 - output[:, t] ** 2
                )
            else:
                a_jac_t_mat_prod[:, :, t] = einsum(
                    "vnh,nh->vnh",
                    mat[:, :, t]
                    + einsum(
                        "vng,gh->vnh",
                        a_jac_t_mat_prod[:, :, t + 1],
                        weight_hh_l0,
                    ),
                    1 - output[:, t] ** 2,
                )
        return a_jac_t_mat_prod

    def _jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)
        return einsum(
            f"vnth,hk->v{'nt' if module.batch_first else 'tn'}k",
            self._a_jac_t_mat_prod(
                module,
                module.weight_hh_l0,
                mat,
                subsampling,
            ),
            module.weight_ih_l0,
        )

    def _jac_mat_prod(
        self, module: RNN, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        self._check_parameters(module)
        H: int = module.hidden_size
        V, N, T, _ = mat.shape
        _jac_mat_prod: Tensor = zeros(V, N, T, H, device=mat.device, dtype=mat.dtype)
        for t in range(T):
            if t == 0:
                _jac_mat_prod[:, :, t] = einsum(
                    "nh,hi,vni->vnh",
                    1 - module.output[:, t] ** 2,
                    module.weight_ih_l0,
                    mat[:, :, t],
                )
            else:
                _jac_mat_prod[:, :, t] = einsum(
                    "nh,vnh->vnh",
                    1 - module.output[:, t] ** 2,
                    einsum(
                        "hi,vni->vnh",
                        module.weight_ih_l0,
                        mat[:, :, t],
                    )
                    + einsum(
                        "hk,vnk->vnh",
                        module.weight_hh_l0,
                        _jac_mat_prod[:, :, t - 1],
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
        subsampling: List[int] = None,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. bias_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.
            subsampling: Indices of active samples. Defaults to ``None`` (all samples).

        Returns:
            product
        """
        self._check_parameters(module)
        if sum_batch:
            dim: List[int] = [1, 2]
        else:
            dim: int = 2
        return self._a_jac_t_mat_prod(
            module,
            module.weight_hh_l0,
            mat,
            subsampling,
        ).sum(dim=dim)

    def _bias_hh_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. bias_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.
            subsampling: Indices of active samples. Defaults to ``None`` (all samples).

        Returns:
            product
        """
        return self._bias_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, mat, sum_batch=sum_batch, subsampling=subsampling
        )

    def _weight_ih_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. weight_ih_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.
            subsampling: Indices of active samples. Defaults to ``None`` (all samples).

        Returns:
            product
        """
        self._check_parameters(module)
        return einsum(
            f"vnth,ntj->v{'' if sum_batch else 'n'}hj",
            self._a_jac_t_mat_prod(module, module.weight_hh_l0, mat, subsampling),
            subsample(module.input0, dim=0, subsampling=subsampling),
        )

    def _weight_hh_l0_jac_t_mat_prod(
        self,
        module: RNN,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Apply transposed Jacobian of the output w.r.t. weight_hh_l0.

        Args:
            module: extended module
            g_inp: input gradient
            g_out: output gradient
            mat: matrix to multiply
            sum_batch: Whether to sum along batch axis. Defaults to True.
            subsampling: Indices of active samples. Defaults to ``None`` (all samples).

        Returns:
            product
        """
        self._check_parameters(module)
        _, N, _, H = mat.shape
        output = subsample(module.output, dim=0, subsampling=subsampling)
        single_step = zeros(N, 1, H, device=mat.device, dtype=mat.dtype)
        output_shifted = cat([single_step, output[:, :-1]], dim=1)
        return einsum(
            f"vnth,ntk->v{'' if sum_batch else 'n'}hk",
            self._a_jac_t_mat_prod(module, module.weight_hh_l0, mat, subsampling),
            output_shifted,
        )

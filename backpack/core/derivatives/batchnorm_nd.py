"""Contains derivatives for BatchNorm."""
from typing import Tuple, Union

from torch import Size, Tensor, einsum
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class BatchNormNdDerivatives(BaseParameterDerivatives):
    """Derivatives for BatchNorm."""

    def __init__(self, n_dim: int):
        """Initialization.

        Args:
            n_dim: number of dimensions
        """
        self.n_dim = n_dim

    def _check_parameters(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> None:
        pass

    def hessian_is_zero(self) -> bool:
        """Whether hessian is zero.

        Returns:
            whether hessian is zero
        """
        return False

    def hessian_is_diagonal(self) -> bool:
        """Whether hessian is diagonal.

        Returns:
            whether hessian is diagonal
        """
        return False

    def _jac_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        return self._jac_t_mat_prod(module, g_inp, g_out, mat)

    def _get_normalized_input_and_var(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> Tuple[Tensor, Tensor]:
        _n_axis: int = self._get_n_axis(module)
        dim: Tuple[int] = {
            0: (0,),
            1: (0, 2),
            2: (0, 2, 3),
            3: (0, 2, 3, 4),
        }[_n_axis]
        input: Tensor = module.input0
        mean: Tensor = input.mean(dim=dim)
        var: Tensor = input.var(dim=dim, unbiased=False)
        mean_expanded: Tensor = {
            0: mean[None, :],
            1: mean[None, :, None],
            2: mean[None, :, None, None],
            3: mean[None, :, None, None, None],
        }[_n_axis]
        var_expanded: Tensor = {
            0: var[None, :],
            1: var[None, :, None],
            2: var[None, :, None, None],
            3: var[None, :, None, None, None],
        }[_n_axis]
        return (input - mean_expanded) / (var_expanded + module.eps).sqrt(), var

    def _jac_t_mat_prod(
        self,
        module: BatchNorm1d,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        self._check_parameters(module)
        _n_dim: int = self._get_n_axis(module)
        denominator: int = self._get_denominator(module)
        x_hat, var = self._get_normalized_input_and_var(module)
        ivar = 1.0 / (var + module.eps).sqrt()

        equation = {
            0: "vni,i->vni",
            1: "vnil,i->vnil",
            2: "vnihw,i->vnihw",
            3: "vnidhw,i->vnidhw",
        }[_n_dim]
        dx_hat: Tensor = einsum(equation, mat, module.weight)
        jac_t_mat = denominator * dx_hat
        _axis_sum: Tuple[int] = {
            0: (1,),
            1: (1, 3),
            2: (1, 3, 4),
            3: (1, 3, 4, 5),
        }[_n_dim]
        jac_t_mat -= dx_hat.sum(
            _axis_sum,
            keepdim=True,
        ).expand_as(jac_t_mat)
        equation = {
            0: "ni,vsi,si->vni",
            1: "nil,vsix,six->vnil",
            2: "nihw,vsixy,sixy->vnihw",
            3: "nidhw,vsixyz,sixyz->vnidhw",
        }[_n_dim]
        jac_t_mat -= einsum(equation, x_hat, dx_hat, x_hat)
        equation = {
            0: "vni,i->vni",
            1: "vnil,i->vnil",
            2: "vnihw,i->vnihw",
            3: "vnidhw,i->vnidhw",
        }[_n_dim]
        jac_t_mat = einsum(equation, jac_t_mat, ivar / denominator)
        return jac_t_mat

    @staticmethod
    def _get_denominator(module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]) -> int:
        shape_input: Size = module.input0.shape
        denominator: int = shape_input[0]
        for i in range(2, len(shape_input)):
            denominator *= shape_input[i]
        return denominator

    @staticmethod
    def _get_n_axis(module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]) -> int:
        return len(module.input0.shape) - 2


class BatchNorm1dDerivatives(BatchNormNdDerivatives):
    """Derivatives for BatchNorm1d."""

    def __init__(self):
        """Initialization."""
        super().__init__(n_dim=1)


class BatchNorm2dDerivatives(BatchNormNdDerivatives):
    """Derivatives for BatchNorm2d."""

    def __init__(self):
        """Initialization."""
        super().__init__(n_dim=2)


class BatchNorm3dDerivatives(BatchNormNdDerivatives):
    """Derivatives for BatchNorm3d."""

    def __init__(self):
        """Initialization."""
        super().__init__(n_dim=3)

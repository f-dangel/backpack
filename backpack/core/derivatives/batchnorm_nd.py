"""Contains derivatives for BatchNorm."""
from typing import List, Tuple, Union

from torch import Size, Tensor, einsum
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class BatchNormNdDerivatives(BaseParameterDerivatives):
    """Derivatives for BatchNorm.

    If training=False: saved statistics are used.
    If training=True: statistics of current batch are used.

    Index convention:
    n: batch axis
    c: category axis
    {empty}/l/hw/dhw: dimension axis for 0/1/2/3-dimensions

    As a starting point, see these references:
    https://kevinzakka.github.io/2016/09/14/batch_normalization/
    https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
    """

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
        if module.training:
            mean: Tensor = input.mean(dim=dim)
            var: Tensor = input.var(dim=dim, unbiased=False)
        else:
            mean: Tensor = module.running_mean
            var: Tensor = module.running_var
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
        if module.training:
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
        else:
            equation = {
                0: "c,vnc->vnc",
                1: "c,vncl->vncl",
                2: "c,vnchw->vnchw",
                3: "c,vncdhw->vncdhw",
            }[_n_dim]
            return einsum(
                equation,
                ((module.running_var + module.eps) ** (-0.5)) * module.weight,
                mat,
            )

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

    def _weight_jac_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        _n_dim: int = self._get_n_axis(module)
        x_hat, _ = self._get_normalized_input_and_var(module)
        equation: str = {
            0: "nc,vc->vnc",
            1: "ncl,vc->vncl",
            2: "nchw,vc->vnchw",
            3: "ncdhw,vc->vncdhw",
        }[_n_dim]
        return einsum(equation, x_hat, mat)

    def _weight_jac_t_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        _n_dim: int = self._get_n_axis(module)
        x_hat, _ = self._get_normalized_input_and_var(module)
        equation = {
            0: f"vnc,nc->v{'' if sum_batch else 'n'}c",
            1: f"vncl,ncl->v{'' if sum_batch else 'n'}c",
            2: f"vnchw,nchw->v{'' if sum_batch else 'n'}c",
            3: f"vncdhw,ncdhw->v{'' if sum_batch else 'n'}c",
        }[_n_dim]
        return einsum(equation, mat, x_hat)

    def _bias_jac_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        _n_axis = self._get_n_axis(module)
        out = mat.unsqueeze(1)
        for _ in range(_n_axis):
            out = out.unsqueeze(-1)
        dim_expand: List[int] = [-1, module.input0.shape[0], -1]
        for n in range(_n_axis):
            dim_expand.append(module.input0.shape[2 + n])
        return out.expand(*dim_expand)

    def _bias_jac_t_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        _n_dim: int = self._get_n_axis(module)
        axis_sum: Tuple[int] = {
            0: (1,) if sum_batch else None,
            1: (1, 3) if sum_batch else (3,),
            2: (1, 3, 4) if sum_batch else (3, 4),
            3: (1, 3, 4, 5) if sum_batch else (3, 4, 5),
        }[_n_dim]
        return mat if axis_sum is None else mat.sum(dim=axis_sum)


class BatchNorm1dDerivatives(BatchNormNdDerivatives):
    """Derivatives for BatchNorm1d."""

    def __init__(self):
        """Initialization."""
        super().__init__(n_dim=1)

    def _residual_mat_prod(
        self,
        module: BatchNorm1d,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        """Multiply with BatchNorm1d residual-matrix.

        Paul Fischer (GitHub: @paulkogni) contributed this code during a research
        project in winter 2019.

        Details are described in

        - `TODO: Add tech report title`
          <TODO: Wait for tech report upload>_
          by Paul Fischer, 2020.

        Args:
            module: module
            g_inp: input gradients
            g_out: output gradients
            mat: matrix to multiply

        Returns:
            product
        """
        N = module.input0.size(0)
        x_hat, var = self._get_normalized_input_and_var(module)
        gamma = module.weight
        eps = module.eps

        factor = gamma / (N * (var + eps))

        sum_127 = einsum("nc,vnc->vc", x_hat, mat)
        sum_24 = einsum("nc->c", g_out[0])
        sum_3 = einsum("nc,vnc->vc", g_out[0], mat)
        sum_46 = einsum("vnc->vc", mat)
        sum_567 = einsum("nc,nc->c", x_hat, g_out[0])

        r_mat = -einsum("nc,vc->vnc", g_out[0], sum_127)
        r_mat += (1.0 / N) * einsum("c,vc->vc", sum_24, sum_127).unsqueeze(1).expand(
            -1, N, -1
        )
        r_mat -= einsum("nc,vc->vnc", x_hat, sum_3)
        r_mat += (1.0 / N) * einsum("nc,c,vc->vnc", x_hat, sum_24, sum_46)

        r_mat -= einsum("vnc,c->vnc", mat, sum_567)
        r_mat += (1.0 / N) * einsum("c,vc->vc", sum_567, sum_46).unsqueeze(1).expand(
            -1, N, -1
        )
        r_mat += (3.0 / N) * einsum("nc,vc,c->vnc", x_hat, sum_127, sum_567)

        return einsum("c,vnc->vnc", factor, r_mat)


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

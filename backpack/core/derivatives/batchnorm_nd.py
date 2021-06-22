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

    Links to PyTorch docs:
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html

    As a starting point for derivative computation, see these references:
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

    def _jac_t_mat_prod(
        self,
        module: BatchNorm1d,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        self._check_parameters(module)
        N: int = self._get_n_axis(module)
        if module.training:
            denominator: int = self._get_denominator(module)
            x_hat, var = self._get_normalized_input_and_var(module)
            ivar = 1.0 / (var + module.eps).sqrt()

            equation = {
                0: "vnc,c->vnc",
                1: "vncl,c->vncl",
                2: "vnchw,c->vnchw",
                3: "vncdhw,c->vncdhw",
            }[N]
            dx_hat: Tensor = einsum(equation, mat, module.weight)
            jac_t_mat = denominator * dx_hat
            jac_t_mat -= dx_hat.sum(
                self._get_free_axes(module),
                keepdim=True,
            ).expand_as(jac_t_mat)
            equation = {
                0: "nc,vmc,mc->vnc",
                1: "ncl,vmcx,mcx->vncl",
                2: "nchw,vmcxy,mcxy->vnchw",
                3: "ncdhw,vmcxyz,mcxyz->vncdhw",
            }[N]
            jac_t_mat -= einsum(equation, x_hat, dx_hat, x_hat)
            equation = {
                0: "vnc,c->vnc",
                1: "vncl,c->vncl",
                2: "vnchw,c->vnchw",
                3: "vncdhw,c->vncdhw",
            }[N]
            jac_t_mat = einsum(equation, jac_t_mat, ivar / denominator)
            return jac_t_mat
        else:
            equation = {
                0: "c,vnc->vnc",
                1: "c,vncl->vncl",
                2: "c,vnchw->vnchw",
                3: "c,vncdhw->vncdhw",
            }[N]
            return einsum(
                equation,
                ((module.running_var + module.eps) ** (-0.5)) * module.weight,
                mat,
            )

    def _weight_jac_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        x_hat, _ = self._get_normalized_input_and_var(module)
        equation: str = {
            0: "nc,vc->vnc",
            1: "ncl,vc->vncl",
            2: "nchw,vc->vnchw",
            3: "ncdhw,vc->vncdhw",
        }[self._get_n_axis(module)]
        return einsum(equation, x_hat, mat)

    def _weight_jac_t_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        x_hat, _ = self._get_normalized_input_and_var(module)
        equation = {
            0: f"vnc,nc->v{'' if sum_batch else 'n'}c",
            1: f"vncl,ncl->v{'' if sum_batch else 'n'}c",
            2: f"vnchw,nchw->v{'' if sum_batch else 'n'}c",
            3: f"vncdhw,ncdhw->v{'' if sum_batch else 'n'}c",
        }[self._get_n_axis(module)]
        return einsum(equation, mat, x_hat)

    def _bias_jac_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        out = self._unsqueeze_free_axis(module, mat, 1)
        dim_expand: List[int] = [-1, module.input0.shape[0], -1]
        for n in range(self._get_n_axis(module)):
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
        axis_sum: Tuple[int] = self._get_free_axes(module, with_batch_axis=sum_batch)
        return mat.sum(dim=axis_sum) if axis_sum else mat

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

        `HESSIAN BACKPROPAGATION FOR BATCHNORM`
        <https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/personen/alumni/everyone-else/>
        <TODO: Wait for tech report upload and add exact link>_
        by Paul Fischer, 2020.

        Args:
            module: module
            g_inp: input gradients
            g_out: output gradients
            mat: matrix to multiply

        Returns:
            product

        Raises:
            NotImplementedError: if used with a not supported mode or input
        """  # noqa: B950
        self._check_parameters(module)
        if module.training is False:
            raise NotImplementedError("residual_mat_prod works only for training mode.")
        if module.input0.dim() != 2:
            raise NotImplementedError(
                "residual_mat_prod is implemented only for 0 dimensions. "
                "If you need more dimension make a feature request."
            )

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

    ###############################################################
    #                  HELPER FUNCTIONS                         ###
    ###############################################################
    def _get_normalized_input_and_var(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> Tuple[Tensor, Tensor]:
        input: Tensor = module.input0
        if module.training:
            dim: Tuple[int] = self._get_free_axes(module, index_batch=0)
            mean: Tensor = input.mean(dim=dim)
            var: Tensor = input.var(dim=dim, unbiased=False)
        else:
            mean: Tensor = module.running_mean
            var: Tensor = module.running_var
        mean: Tensor = self._unsqueeze_free_axis(module, mean, 0)
        var_expanded: Tensor = self._unsqueeze_free_axis(module, var, 0)
        return (input - mean) / (var_expanded + module.eps).sqrt(), var

    def _get_denominator(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> int:
        shape_input: Size = module.input0.shape
        free_axes: Tuple[int] = self._get_free_axes(module, index_batch=0)
        denominator: int = 1
        for index in free_axes:
            denominator *= shape_input[index]
        return denominator

    @staticmethod
    def _get_n_axis(module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]) -> int:
        return module.input0.dim() - 2

    def _unsqueeze_free_axis(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        tensor: Tensor,
        index_batch: int,
    ) -> Tensor:
        out = tensor.unsqueeze(index_batch)
        for _ in range(self._get_n_axis(module)):
            out = out.unsqueeze(-1)
        return out

    def _get_free_axes(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        with_batch_axis: bool = True,
        index_batch: int = 1,
    ) -> Tuple[int]:
        free_axes: List[int] = []
        if with_batch_axis:
            free_axes.append(index_batch)
        for n in range(self._get_n_axis(module)):
            free_axes.append(index_batch + n + 2)
        return tuple(free_axes)


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

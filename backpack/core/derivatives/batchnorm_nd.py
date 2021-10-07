"""Contains derivatives for BatchNorm."""
from typing import List, Tuple, Union

from torch import Size, Tensor, einsum
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class BatchNormNdDerivatives(BaseParameterDerivatives):
    """Derivatives for BatchNorm1d, 2d and 3d.

    If training=False: saved statistics are used.
    If training=True: statistics of current batch are used.

    Index convention:
    n: batch axis
    c: category axis
    {empty}/l/hw/dhw: dimension axis for 0/1/2/3-dimensions (alternatively using xyz)
    ...: usually for the remaining dimension axis (same as dhw)

    Links to PyTorch docs:
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html

    As a starting point for derivative computation, see these references:
    https://kevinzakka.github.io/2016/09/14/batch_normalization/
    https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
    """

    def _check_parameters(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> None:
        if module.affine is False:
            raise NotImplementedError("Only implemented for affine=True")
        if module.track_running_stats is False:
            raise NotImplementedError("Only implemented for track_running_stats=True")

    def hessian_is_zero(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> bool:
        """Whether hessian is zero.

        Args:
            module: current module to evaluate

        Returns:
            whether hessian is zero
        """
        return not module.training

    def hessian_is_diagonal(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> bool:
        """Whether hessian is diagonal.

        Args:
            module: current module to evaluate

        Returns:
            whether hessian is diagonal

        Raises:
            NotImplementedError: if module is in evaluation mode
        """
        if module.training:
            return False
        else:
            raise NotImplementedError(
                "hessian_is_diagonal is not tested for BatchNorm. "
                "Create an issue if you need it."
            )

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
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)
        N: int = self._get_n_axis(module)
        if module.training:

            if subsampling is not None:
                raise NotImplementedError(
                    "BatchNorm VJP sub-sampling is not defined in train mode."
                )

            denominator: int = self._get_denominator(module)
            x_hat, var = self._get_normalized_input_and_var(module)
            ivar = 1.0 / (var + module.eps).sqrt()

            dx_hat: Tensor = einsum("vnc...,c->vnc...", mat, module.weight)
            jac_t_mat = denominator * dx_hat
            jac_t_mat -= dx_hat.sum(
                self._get_free_axes(module),
                keepdim=True,
            ).expand_as(jac_t_mat)
            spatial_dims = "xyz"[:N]
            jac_t_mat -= einsum(
                f"nc...,vmc{spatial_dims},mc{spatial_dims}->vnc...",
                x_hat,
                dx_hat,
                x_hat,
            )
            jac_t_mat = einsum("vnc...,c->vnc...", jac_t_mat, ivar / denominator)
            return jac_t_mat
        else:
            return einsum(
                "c,vnc...->vnc...",
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
        return einsum("nc...,vc->vnc...", x_hat, mat)

    def _weight_jac_t_mat_prod(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        x_hat, _ = self._get_normalized_input_and_var(module)
        x_hat = subsample(x_hat, subsampling=subsampling)

        equation = f"vnc...,nc...->v{'' if sum_batch else 'n'}c"
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
        subsampling: List[int] = None,
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
        <https://uni-tuebingen.de/securedl/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2MjU0ODY0NzAsImV4cCI6MTYyNTU3NjQ2NCwidXNlciI6MCwiZ3JvdXBzIjpbMCwtMV0sImZpbGUiOiJmaWxlYWRtaW5cL1VuaV9UdWViaW5nZW5cL0Zha3VsdGFldGVuXC9NYXROYXRcL0ZhY2hiZXJlaWNoZVwvSW5mb3JtYXRpa1wvTGVocnN0dWVobGVcL01ldGhNYXNjaExlcm5cL0Rva3VtZW50ZVwvVGhlc2VzXC9BYnNjaGx1c3NiZXJpY2h0X0Zpc2NoZXIucGRmIiwicGFnZSI6MTczNDc1fQ.xMe4WmTsyak9J-C7iTxGqTdpYMxMxtGfouyAZgW158I/Abschlussbericht_Fischer.pdf>
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
        """Unsqueezes the free dimensions.

        This function is useful to avoid broadcasting.
        Also useful when applying .expand(self._get_free_axes()) afterwards.

        Args:
            module: extended module
            tensor: the tensor to operate on
            index_batch: the batch axes index

        Returns:
            tensor with the free dimensions unsqueezed.
        """
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

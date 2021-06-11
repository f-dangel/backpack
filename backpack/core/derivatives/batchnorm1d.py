from typing import Tuple
from warnings import warn

import torch
from torch import Tensor, diag_embed, diagonal, einsum, ones, sqrt, zeros
from torch.nn import BatchNorm1d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class BatchNorm1dDerivatives(BaseParameterDerivatives):
    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return False

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self._jac_t_mat_prod(module, g_inp, g_out, mat)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """
        Note:
        -----
        The Jacobian is *not independent* among the batch dimension, i.e.
        D z_i = D z_i(x_1, ..., x_B).

        This structure breaks the computation of the GGN diagonal,
        for curvature-matrix products it should still work.

        References:
        -----------
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
        """
        assert module.affine is True

        N = module.input0.size(0)
        x_hat, var = self._get_normalized_input_and_var(module)
        ivar = 1.0 / (var + module.eps).sqrt()

        dx_hat = einsum("vni,i->vni", (mat, module.weight))

        jac_t_mat = N * dx_hat
        jac_t_mat -= dx_hat.sum(1).unsqueeze(1).expand_as(jac_t_mat)
        jac_t_mat -= einsum("ni,vsi,si->vni", (x_hat, dx_hat, x_hat))
        jac_t_mat = einsum("vni,i->vni", (jac_t_mat, ivar / N))

        return jac_t_mat

    @classmethod
    def _get_normalized_input_and_var(cls, module: BatchNorm1d):
        dim: Tuple[int] = (0, 2) if cls._has_l_axis(module) else (0,)
        input: Tensor = module.input0
        mean: Tensor = input.mean(dim=dim)
        var: Tensor = input.var(dim=dim, unbiased=False)
        mean_expanded = (
            mean[None, :, None]
            if cls._has_l_axis(module)
            else mean[None, :]
        )
        var_expanded = (
            var[None, :, None]
            if cls._has_l_axis(module)
            else var[None, :]
        )
        return (input - mean_expanded) / (var_expanded + module.eps).sqrt(), var

    def _jac_t_mat_prod_alternative(
        self,
        module: BatchNorm1d,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        _has_l_axis = self._has_l_axis(module)
        if _has_l_axis:
            denominator: int = mat.shape[1] * mat.shape[3]
        else:
            denominator: int = mat.shape[1]
        x_hat, var = self._get_normalized_input_and_var(module)
        ivar = 1.0 / (var + module.eps).sqrt()

        equation = "vnil,i->vnil" if _has_l_axis else "vni,i->vni"
        dx_hat: Tensor = einsum(equation, mat, module.weight)
        jac_t_mat = denominator * dx_hat
        jac_t_mat -= dx_hat.sum(
            (1, 3) if _has_l_axis else (1,),
            keepdim=True,
        ).expand_as(jac_t_mat)
        equation = "nil,vsik,sik->vnil" if _has_l_axis else "ni,vsi,si->vni"
        jac_t_mat -= einsum(equation, x_hat, dx_hat, x_hat)
        equation = "vnil,i->vnil" if _has_l_axis else "vni,i->vni"
        jac_t_mat = einsum(equation, jac_t_mat, ivar / denominator)
        return jac_t_mat

    @classmethod
    def _has_l_axis(cls, module: BatchNorm1d) -> bool:
        _dim = module.input0.dim()
        if _dim == 3:
            return True
        elif _dim == 2:
            return False
        else:
            raise NotImplemented(
                f"Can't handle input with {_dim} dimensions on module BatchNorm1d."
            )

    def _residual_mat_prod(self, module, g_inp, g_out, mat):
        """Multiply with BatchNorm1d residual-matrix.

        Paul Fischer (GitHub: @paulkogni) contributed this code during a research
        project in winter 2019.

        Details are described in

        - `TODO: Add tech report title`
          <TODO: Wait for tech report upload>_
          by Paul Fischer, 2020.
        """
        N = module.input0.size(0)
        x_hat, var = self._get_normalized_input_and_var(module)
        gamma = module.weight
        eps = module.eps

        factor = gamma / (N * (var + eps))

        sum_127 = einsum("nc,vnc->vc", (x_hat, mat))
        sum_24 = einsum("nc->c", g_out[0])
        sum_3 = einsum("nc,vnc->vc", (g_out[0], mat))
        sum_46 = einsum("vnc->vc", mat)
        sum_567 = einsum("nc,nc->c", (x_hat, g_out[0]))

        r_mat = -einsum("nc,vc->vnc", (g_out[0], sum_127))
        r_mat += (1.0 / N) * einsum("c,vc->vc", (sum_24, sum_127)).unsqueeze(1).expand(
            -1, N, -1
        )
        r_mat -= einsum("nc,vc->vnc", (x_hat, sum_3))
        r_mat += (1.0 / N) * einsum("nc,c,vc->vnc", (x_hat, sum_24, sum_46))

        r_mat -= einsum("vnc,c->vnc", (mat, sum_567))
        r_mat += (1.0 / N) * einsum("c,vc->vc", (sum_567, sum_46)).unsqueeze(1).expand(
            -1, N, -1
        )
        r_mat += (3.0 / N) * einsum("nc,vc,c->vnc", (x_hat, sum_127, sum_567))

        return einsum("c,vnc->vnc", (factor, r_mat))

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        x_hat, _ = self._get_normalized_input_and_var(module)
        return einsum("ni,vi->vni", (x_hat, mat))

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch):
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
        x_hat, _ = self._get_normalized_input_and_var(module)
        equation = "vni,ni->v{}i".format("" if sum_batch is True else "n")
        operands = [mat, x_hat]
        return einsum(equation, operands)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        N = module.input0.size(0)
        return mat.unsqueeze(1).repeat(1, N, 1)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
            return mat
        else:
            N_axis = 1
            return mat.sum(N_axis)

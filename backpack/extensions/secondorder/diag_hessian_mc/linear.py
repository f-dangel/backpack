"""Contains module extension of ``nn.Linear`` to approximate the Hessian diagonal."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from torch import Tensor, einsum
from torch.nn import Module

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.module_extension import ModuleExtension

if TYPE_CHECKING:
    from backpack.extensions.secondorder.diag_hessian_mc import DiagHessianMC


class DiagHMCLinear(ModuleExtension):
    """Monte-Carlo estimate of the Hessian diagonal for ``nn.Linear``."""

    def backpropagate(
        self,
        extension: DiagHessianMC,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        mc_samples = extension.get_num_mc_samples()
        T_output, U_output = bpQuantities["T"], bpQuantities["U"]

        input0 = module.input0
        v_input0 = extension.get_random_vector(input0.shape, input0.device)
        weight = module.weight
        v_weight = extension.get_random_vector(weight.shape, weight.device)
        bias = module.bias
        v_bias = extension.get_random_vector(bias.shape, bias.device)

        derivatives = LinearDerivatives()

        # weight
        Mv_weight = einsum("ni,vnj->vij", g_out[0], v_input0)
        T_weight = Mv_weight + derivatives.param_mjp(
            "weight", module, g_inp, g_out, T_output, sum_batch=True
        )
        U_weight = v_weight + derivatives.param_mjp(
            "weight", module, g_inp, g_out, U_output, sum_batch=True
        )
        weight.diag_h_mc = einsum("vij,vij->ij", U_weight, T_weight) / mc_samples

        # bias
        T_bias = derivatives.param_mjp(
            "bias", module, g_inp, g_out, T_output, sum_batch=True
        )
        U_bias = v_bias + derivatives.param_mjp(
            "bias", module, g_inp, g_out, U_output, sum_batch=True
        )
        bias.diag_h_mc = einsum("vb,vb->b", U_bias, T_bias) / mc_samples

        # output
        Mv_input = einsum("nj,vjk->vnk", g_out[0], v_weight)

        return {
            "T": Mv_input + derivatives.jac_t_mat_prod(module, g_inp, g_out, T_output),
            "U": v_input0 + derivatives.jac_t_mat_prod(module, g_inp, g_out, U_output),
        }

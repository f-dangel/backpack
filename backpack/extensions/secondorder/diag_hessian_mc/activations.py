"""Module extension of ``nn`` activations to approximate the Hessian diagonal."""


from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from torch import Tensor, einsum
from torch.nn import Module

from backpack.core import derivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.extensions.module_extension import ModuleExtension

if TYPE_CHECKING:
    from backpack.extensions.secondorder.diag_hessian_mc import DiagHessianMC


class DiagHMCSigmoid(ModuleExtension):
    def backpropagate(
        self,
        extension: DiagHessianMC,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        T_output, U_output = bpQuantities["T"], bpQuantities["U"]

        input0 = module.input0
        v_input0 = extension.get_random_vector(input0.shape, input0.device)

        derivatives = SigmoidDerivatives()

        d2f = derivatives.d2f(module, g_inp, g_out)
        Mv_input = einsum("...,...,v...->v...", d2f, g_out[0], v_input0)

        T_input = Mv_input + derivatives.jac_t_mat_prod(module, g_inp, g_out, T_output)
        U_input = v_input0 + derivatives.jac_t_mat_prod(module, g_inp, g_out, U_output)

        return {"T": T_input, "U": U_input}

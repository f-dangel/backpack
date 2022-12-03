"""Module extensions of ``nn`` loss functions to approximate the Hessian diagonal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.extensions.module_extension import ModuleExtension

if TYPE_CHECKING:
    from backpack.extensions.secondorder.diag_hessian_mc import DiagHessianMC


class DiagHMCMSELoss(ModuleExtension):
    """Monte-Carlo estimate of the Hessian diagonal for ``nn.MSELoss``."""

    def backpropagate(
        self,
        extension: DiagHessianMC,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        input0 = module.input0
        v = extension.get_random_vector(input0.shape, input0.device)

        if module.reduction == "sum":
            return {"T": 2 * v, "U": v}
        elif module.reduction == "mean":
            normalization = input0.numel()
            return {"T": 2 * v / normalization, "U": v}
        else:
            raise ValueError

import torch
import math
import torch.nn.functional as F
from ..context import CTX
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_H


class DiagHMSELoss(BackpropExtension):
    def __init__(self):
        super().__init__(torch.nn.MSELoss, DIAG_H, req_inputs=[0])

    # TODO: Reuse code in ..diaggn.mseloss
    def apply(self, module, grad_input, grad_output):
        if not len(module.input0.shape) == 2:
            raise ValueError(
                "Only 2D inputs are currently supported for MSELoss.")

        sqrt_h_in = torch.diag_embed(
            math.sqrt(2) * torch.ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_h_in /= math.sqrt(module.input0.shape[0])

        CTX._backpropagated_sqrt_h = [sqrt_h_in]
        CTX._backpropagated_sqrt_h_signs = [1.]


EXTENSIONS = [DiagHMSELoss()]

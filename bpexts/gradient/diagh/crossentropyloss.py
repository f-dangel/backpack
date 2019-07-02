import torch
import math
import torch.nn.functional as F
from ..context import CTX
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_H


class DiagGGNCrossEntropyLoss(BackpropExtension):
    def __init__(self):
        super().__init__(
            torch.nn.CrossEntropyLoss, DIAG_H, req_inputs=[0], req_output=True)

    # TODO: Reuse code in ..diaggn.crossentropyloss
    def apply(self, module, grad_input, grad_output):
        probs = F.softmax(module.input0, dim=1)
        tau = torch.sqrt(probs)
        Id = torch.diag_embed(torch.ones_like(probs))
        Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
        sqrt_h_in = einsum('ni,nij->nij', tau, Id_tautau)

        if module.reduction is "mean":
            sqrt_h_in /= math.sqrt(module.input0.shape[0])

        CTX._backpropagated_sqrt_h = [sqrt_h_in]
        CTX._backpropagated_sqrt_h_signs = [1.]


EXTENSIONS = [DiagHCrossEntropyLoss()]

import torch
import math
from ..backpropextension import BackpropExtension
from ..context import CTX


class DiagGGNMSELoss(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.MSELoss, "DIAG_GGN",
            req_inputs=[0]
        )

    def apply(self, module, grad_input, grad_output):
        self.backpropagate_sqrt_ggn(module)

    def backpropagate_sqrt_ggn(self, module):
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

        sqrt_ggn_in = torch.diag_embed(math.sqrt(2) * torch.ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_ggn_in /= math.sqrt(module.input0.shape[0])

        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


EXTENSIONS = [DiagGGNMSELoss()]

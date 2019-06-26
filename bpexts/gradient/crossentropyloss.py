import torch
import torch.nn.functional as F
from torch import einsum
from . import config
from .config import CTX


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_forward_pre_hook(self.store_input)
        self.register_backward_hook(self.compute_first_order_info)

    @staticmethod
    def store_input(module, inputs):
        assert len(inputs) == 2
        module.register_buffer('input', inputs[0].clone().detach())

    @staticmethod
    def compute_first_order_info(module, grad_input, grad_output):
        if CTX.is_active(config.DIAG_GGN):
            CTX._backpropagated_sqrt_ggn = module.sqrt_hessians()

    def sqrt_hessians(self):
        probs = F.softmax(self.input, dim=1)
        tau = torch.sqrt(probs)
        Id = torch.diag_embed(torch.ones_like(self.probs))
        Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
        return einsum('ni,nij->nij', tau, Id_tautau)

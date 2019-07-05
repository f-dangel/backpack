import torch.nn
from ..jmp.maxpool2d import jac_mat_prod
from .base import DiagGGNBase


class DiagGGNMaxpool2d(DiagGGNBase):

    def get_module(self):
        return torch.nn.MaxPool2d

    def jac_mat_prod(self, module, grad_input, grad_output, sqrt_ggn_out):
        return jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)


EXTENSIONS = [DiagGGNMaxpool2d()]

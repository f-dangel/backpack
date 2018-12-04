"""Hessian backpropagation for a composition of ReLU and linear layer."""

from .relu import HBPReLU
from .combined import HBPCompositionActivationLinear


class HBPReLULinear(HBPCompositionActivationLinear):
    """ReLU linear layer with HBP in KFRA style."""
    nonlinear_cls = HBPReLU

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(self.nonlinear_cls, in_features, out_features,
                         bias=bias)

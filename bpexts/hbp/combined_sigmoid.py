"""Hessian backpropagation for a composition of Sigmoid and linear layer."""

from .sigmoid import HBPSigmoid
from .combined import HBPCompositionActivationLinear


class HBPSigmoidLinear(HBPCompositionActivationLinear):
    """Sigmoid linear layer with HBP in BDA-PCH style."""
    nonlinear_cls = HBPSigmoid

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(self.nonlinear_cls, in_features, out_features,
                         bias=bias)

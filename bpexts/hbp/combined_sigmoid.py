"""Hessian backpropagation for a composition of Sigmoid and linear layer."""

from .sigmoid import HBPSigmoid
from .combined import HBPCompositionActivationLinear
from ..utils import SigmoidLinear


class HBPSigmoidLinear(HBPCompositionActivationLinear):
    """Sigmoid linear layer with HBP in BDA-PCH style."""
    nonlinear_cls = HBPSigmoid

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, SigmoidLinear):
            raise ValueError(
                "Expecting bpexts.utils.SigmoidLinear, got {}".format(
                    torch_layer.__class__))
        # create instance
        combined = cls(
            in_features=torch_layer.in_features,
            out_features=torch_layer.out_features,
            bias=torch_layer.bias is not None)
        # copy parameters
        combined.linear.weight = torch_layer.weight
        combined.linear.bias = torch_layer.bias
        return combined

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            self.nonlinear_cls, in_features, out_features, bias=bias)

"""Hessian backpropagation for a composition of ReLU and linear layer."""

from .relu import HBPReLU
from .combined import HBPCompositionActivationLinear
from ..utils import ReLULinear


class HBPReLULinear(HBPCompositionActivationLinear):
    """ReLU linear layer with HBP in KFRA style."""
    nonlinear_cls = HBPReLU

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, ReLULinear):
            raise ValueError(
                "Expecting bpexts.utils.ReLULinear, got {}".format(
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

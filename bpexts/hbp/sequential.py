"""Hessian backpropagation implementation of torch.nn.Sequential. and conversion of torch.nn layers to HBP layers."""

from .module import hbp_decorate
# torch layers
from torch.nn import (ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Sequential)
from ..utils import Flatten, SigmoidLinear, ReLULinear
# HBP layers
from .combined_relu import HBPReLULinear
from .combined_sigmoid import HBPSigmoidLinear
from .relu import HBPReLU
from .sigmoid import HBPSigmoid
from .linear import HBPLinear
from .conv2d import HBPConv2d
from .conv2d_recursive import HBPConv2dRecursive
from .maxpool2d import HBPMaxPool2d
from .maxpool2d_recursive import HBPMaxPool2dRecursive
from .flatten import HBPFlatten


class HBPSequential(hbp_decorate(Sequential)):
    """A sequence of HBP modules."""
    # override
    @classmethod
    def from_torch(cls, torch_layer, use_recursive=True):
        if not isinstance(torch_layer, Sequential):
            raise ValueError("Expecting torch.nn.Sequential, got {}".format(
                torch_layer.__class__))
        layers = []
        for mod in torch_layer:
            layers.append(
                convert_torch_to_hbp(mod, use_recursive=use_recursive))
        return cls(*layers)

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def backward_hessian(self,
                         output_hessian,
                         compute_input_hessian=False,
                         modify_2nd_order_terms='none'):
        """Propagate Hessian through the network.

        Starting from the last layer, call `backward_hessian` recursively
        until ending up in the first module.
        """
        out_h = output_hessian
        for idx in reversed(range(len(self))):
            module = self[idx]
            compute_in = True if (idx != 0) else compute_input_hessian
            out_h = module.backward_hessian(
                out_h,
                compute_input_hessian=compute_in,
                modify_2nd_order_terms=modify_2nd_order_terms)
        return out_h


def _supported_conversions(use_recursive):
    """Return supported conversions."""
    return [(ReLU, HBPReLU), (Sigmoid, HBPSigmoid), (Linear, HBPLinear),
            (ReLULinear, HBPReLULinear), (SigmoidLinear, HBPSigmoidLinear),
            (Conv2d, HBPConv2dRecursive if use_recursive else HBPConv2d),
            (MaxPool2d,
             HBPMaxPool2dRecursive if use_recursive else HBPMaxPool2d),
            (Sequential, HBPSequential), (Flatten, HBPFlatten)]


def convert_torch_to_hbp(layer, use_recursive=True):
    """Convert torch layer to corresponding HBP layer.

    Parameters:
    -----------
    use_recursive : bool
        Use recursive layers for convolution and max pooling.
    """
    conversions = _supported_conversions(use_recursive)
    for (torch_cls, hbp_cls) in conversions:
        if isinstance(layer, torch_cls):
            if isinstance(layer, Sequential):
                return hbp_cls.from_torch(layer, use_recursive=use_recursive)
            else:
                return hbp_cls.from_torch(layer)
    _print_conversions()
    raise ValueError("Class {} cannot be converted to HBP.".format(
        layer.__class__))


def _print_conversions():
    """Print all possible conversions."""
    for recursion in [False, True]:
        print("\nSupported conversions (recursion = {}):".format(recursion))
        for torch_cls, hbp_cls in _supported_conversions(recursion):
            print("{:>20}\t->{:>25}".format(torch_cls.__name__,
                                            hbp_cls.__name__))

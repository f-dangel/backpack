"""Curvature-vector products for sequences of modules and conversion of torch.nn layers to CVP layers."""

from ..hbp.module import hbp_decorate
# torch layers
from torch.nn import (ReLU, Sigmoid, Tanh, Linear, Conv2d, MaxPool2d,
                      Sequential, ZeroPad2d, CrossEntropyLoss)
from ..utils import Flatten, Conv2dSame, MaxPool2dSame, same_padding2d
# CVP layers
from .relu import CVPReLU
from .sigmoid import CVPSigmoid
from .tanh import CVPTanh
from .linear import CVPLinear
from .conv2d import CVPConv2d
from .padding import CVPZeroPad2d
from .maxpool2d import CVPMaxPool2d
from .flatten import CVPFlatten
from .crossentropy import CVPCrossEntropyLoss


class CVPSequential(hbp_decorate(Sequential)):
    """Sequence of modules with recursive Hessian-vector products."""
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Sequential):
            raise ValueError("Expecting torch.nn.Sequential, got {}".format(
                torch_layer.__class__))
        layers = []
        for mod in torch_layer:
            layers.append(convert_torch_to_cvp(mod))
        return cls(*layers)

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def backward_hessian(self,
                         output_hessian,
                         compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        """Propagate Hessian-vector product through the network.

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


class CVPConv2dSame(CVPSequential):
    """2D Convolution with padding same and recursive Hessian-vector
    products."""
    """2d convolution with padding same and support for asymmetric padding."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            CVPZeroPad2d(0),
            CVPConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=True))

    def forward(self, x):
        self._adapt_padding(x)
        out = super().forward(x)
        # remove after proper testing
        if self[1].stride == 1 or self[1].stride == (1, 1):
            if not x.size()[2:] == out.size()[2:]:
                raise ValueError(
                    "Expect same sizes for input and output, got {}, {}".
                    format(x.size(), out.size()))
        return out

    def _adapt_padding(self, x):
        """Adapt the parameters of zero padding depending on input size."""
        assert self[1].dilation == 1 or self[1].dilation == (1, 1)
        padding = same_padding2d(
            input_dim=(x.size(2), x.size(3)),
            kernel_dim=self[1].kernel_size,
            stride_dim=self[1].stride)
        self[0].padding = padding

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Conv2dSame):
            raise ValueError(
                "Expecting bpexts.utils.Conv2dSame, got {}".format(
                    torch_layer.__class__))
        # create instance
        conv2dsame = cls(
            torch_layer[1].in_channels,
            torch_layer[1].out_channels,
            torch_layer[1].kernel_size,
            stride=torch_layer[1].stride,
            dilation=torch_layer[1].dilation,
            groups=torch_layer[1].groups,
            bias=torch_layer[1].bias is not None)
        # copy parameters
        conv2dsame[1].weight = torch_layer[1].weight
        conv2dsame[1].bias = torch_layer[1].bias
        return conv2dsame


class CVPMaxPool2dSame(CVPSequential):
    """2d max pooling with padding same and support for asymmetric padding."""

    def __init__(self,
                 kernel_size,
                 stride=None,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        super().__init__(
            CVPZeroPad2d(0),
            CVPMaxPool2d(
                kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode))

    def forward(self, x):
        self._adapt_padding(x)
        out = super().forward(x)
        # remove after proper testing
        if self[1].stride == 1 or self[1].stride == (1, 1):
            if not x.size()[2:] == out.size()[2:]:
                raise ValueError(
                    "Expect same sizes for input and output, got {}, {}".
                    format(x.size(), out.size()))
        return out

    def _adapt_padding(self, x):
        """Adapt the parameters of zero padding depending on input size."""
        assert self[1].dilation == 1 or self[1].dilation == (1, 1)
        padding = same_padding2d(
            input_dim=(x.size(2), x.size(3)),
            kernel_dim=self[1].kernel_size,
            stride_dim=self[1].stride)
        self[0].padding = padding

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, MaxPool2dSame):
            raise ValueError(
                "Expecting bpexts.utils.MaxPool2dSame, got {}".format(
                    torch_layer.__class__))
        # create instance
        maxpool2dsame = cls(
            torch_layer[1].kernel_size,
            stride=torch_layer[1].stride,
            dilation=torch_layer[1].dilation,
            return_indices=torch_layer[1].return_indices,
            ceil_mode=torch_layer[1].ceil_mode)
        return maxpool2dsame


# supported conversions
conversions = [(ReLU, CVPReLU), (Sigmoid, CVPSigmoid), (Tanh, CVPTanh),
               (Linear, CVPLinear), (Conv2d, CVPConv2d),
               (MaxPool2d, CVPMaxPool2d), (Sequential, CVPSequential),
               (Flatten, CVPFlatten), (MaxPool2dSame, CVPMaxPool2dSame),
               (Conv2dSame, CVPConv2dSame),
               (CrossEntropyLoss, CVPCrossEntropyLoss)]


def convert_torch_to_cvp(layer):
    """Convert torch layer to corresponding CVP layer."""
    for (torch_cls, cvp_cls) in conversions:
        if isinstance(layer, torch_cls):
            return cvp_cls.from_torch(layer)
    _print_conversions()
    raise ValueError("Class {} cannot be converted to CVP.".format(
        layer.__class__))


def _print_conversions():
    """Print all possible conversions."""
    print("Supported conversions:")
    for torch_cls, cvp_cls in conversions:
        print("{}\t->\t{}".format(torch_cls.__name__, cvp_cls.__name__))

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat, Flatten
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, Tanh, Sigmoid, ReLU, CrossEntropyLoss, MSELoss, AvgPool2d, ZeroPad2d
from . import pooling, conv2d, linear, activations, losses, padding, dropout, flatten


class DiagGGN(BackpropExtension):

    def __init__(self):
        super().__init__(
            savefield="diag_ggn",
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.DiagGGNLinear(),
                LinearConcat: linear.DiagGGNLinearConcat(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv2d: conv2d.DiagGGNConv2d(),
                Conv2dConcat: conv2d.DiagGGNConv2dConcat(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
            }
        )

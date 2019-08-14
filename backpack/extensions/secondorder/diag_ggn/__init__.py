from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat, Flatten
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, Tanh, Sigmoid, ReLU, CrossEntropyLoss, MSELoss, AvgPool2d, \
    ZeroPad2d
from . import pooling, conv2d, linear, activations, losses, padding, dropout, flatten


class DiagGGN(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING
    ]

    def __init__(self, loss_hessian_strategy=LossHessianStrategy.EXACT):
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy) +
                "Valid strategies: [{}]".format(
                    self.VALID_LOSS_HESSIAN_STRATEGIES
                )
            )

        self.loss_hessian_strategy = loss_hessian_strategy
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

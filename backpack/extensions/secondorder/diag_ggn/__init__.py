from torch.nn import (AvgPool2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, MSELoss, ReLU, Sigmoid, Tanh, ZeroPad2d)

from backpack.core.layers import Conv2dConcat, Flatten, LinearConcat
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from . import (activations, conv2d, dropout, flatten, linear, losses, padding,
               pooling)


class DiagGGN(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT, LossHessianStrategy.SAMPLING
    ]

    def __init__(self,
                 loss_hessian_strategy=LossHessianStrategy.EXACT,
                 savefield=None):
        if savefield is None:
            savefield = "diag_ggn"
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy) +
                "Valid strategies: [{}]".format(
                    self.VALID_LOSS_HESSIAN_STRATEGIES))

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(savefield=savefield,
                         fail_mode="ERROR",
                         module_exts={
                             MSELoss: losses.DiagGGNMSELoss(),
                             CrossEntropyLoss:
                             losses.DiagGGNCrossEntropyLoss(),
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
                         })


class DiagGGNExact(DiagGGN):
    """Exact diagonal of the GGN"""
    def __init__(self, savefield=None):
        if savefield is None:
            savefield = "diag_ggn_exact"
        super().__init__(loss_hessian_strategy=LossHessianStrategy.EXACT,
                         savefield=savefield)


class DiagGGNMC(DiagGGN):
    """MC-sampled diagonal of the GGN/Fisher."""
    def __init__(self, savefield=None):
        if savefield is None:
            savefield = "diag_ggn_mc"
        super().__init__(loss_hessian_strategy=LossHessianStrategy.SAMPLING,
                         savefield=savefield)

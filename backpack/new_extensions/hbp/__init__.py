from backpack.curvature import Curvature
from backpack.new_extensions.hbp.hbp_options import LossHessianStrategy, BackpropStrategy, ExpectationApproximation
from backpack.newbackpropextension import NewBackpropExtension

from backpack.core.layers import Conv2dConcat, LinearConcat, Flatten
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, Tanh, Sigmoid, ReLU, CrossEntropyLoss, MSELoss, AvgPool2d, ZeroPad2d
from . import pooling, conv2d, linear, activations, losses, padding, dropout, flatten


class HBP(NewBackpropExtension):
    def __init__(self, curv_type, loss_hessian_strategy, backprop_strategy, ea_strategy, savefield="hbp"):
        self.curv_type = curv_type
        self.loss_hessian_strategy = loss_hessian_strategy
        self.backprop_strategy = backprop_strategy
        self.ea_strategy = ea_strategy

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.HBPMSELoss(),
                CrossEntropyLoss: losses.HBPCrossEntropyLoss(),
                Linear: linear.HBPLinear(),
                LinearConcat: linear.HBPLinearConcat(),
                MaxPool2d: pooling.HBPMaxpool2d(),
                AvgPool2d: pooling.HBPAvgPool2d(),
                ZeroPad2d: padding.HBPZeroPad2d(),
                Conv2d: conv2d.HBPConv2d(),
                Conv2dConcat: conv2d.HBPConv2dConcat(),
                Dropout: dropout.HBPDropout(),
                Flatten: flatten.HBPFlatten(),
                ReLU: activations.HBPReLU(),
                Sigmoid: activations.HBPSigmoid(),
                Tanh: activations.HBPTanh(),
            }
        )

    def get_curv_type(self):
        return self.curv_type

    def get_loss_hessian_strategy(self):
        return self.loss_hessian_strategy

    def get_backprop_strategy(self):
        return self.backprop_strategy

    def get_ea_strategy(self):
        return self.ea_strategy


class KFAC(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.SAMPLING,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfac"
        )


class KFRA(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.AVERAGE,
            backprop_strategy=BackpropStrategy.BATCH_AVERAGE,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfra",
        )


class KFLR(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.EXACT,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kflr",
        )

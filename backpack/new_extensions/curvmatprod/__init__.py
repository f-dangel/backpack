from backpack.newbackpropextension import NewBackpropExtension

from backpack.core.layers import Conv2dConcat, LinearConcat, Flatten
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, Tanh, Sigmoid, ReLU, CrossEntropyLoss, MSELoss, AvgPool2d, ZeroPad2d
from . import pooling, conv2d, linear, activations, losses, padding, dropout, flatten


class CMP(NewBackpropExtension):
    def __init__(self, curv_type, savefield="cmp"):
        self.curv_type = curv_type

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.CMPMSELoss(),
                CrossEntropyLoss: losses.CMPCrossEntropyLoss(),
                Linear: linear.CMPLinear(),
                LinearConcat: linear.CMPLinearConcat(),
                MaxPool2d: pooling.CMPMaxpool2d(),
                AvgPool2d: pooling.CMPAvgPool2d(),
                ZeroPad2d: padding.CMPZeroPad2d(),
                Conv2d: conv2d.CMPConv2d(),
                Conv2dConcat: conv2d.CMPConv2dConcat(),
                Dropout: dropout.CMPDropout(),
                Flatten: flatten.CMPFlatten(),
                ReLU: activations.CMPReLU(),
                Sigmoid: activations.CMPSigmoid(),
                Tanh: activations.CMPTanh(),
            }
        )

    def get_curv_type(self):
        return self.curv_type

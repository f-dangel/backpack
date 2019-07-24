from torch.nn import Sigmoid, Tanh, ReLU, Dropout, ZeroPad2d, MaxPool2d, Linear, AvgPool2d, Conv2d
from backpack.core.layers import LinearConcat, Conv2dConcat
from .linear import LinearDerivatives, LinearConcatDerivatives
from .conv2d import Conv2DDerivatives, Conv2DConcatDerivatives
from .avgpool2d import AvgPool2DDerivatives
from .maxpool2d import MaxPool2DDerivatives
from .zeropad2d import ZeroPad2dDerivatives
from .dropout import DropoutDerivatives
from .relu import ReLUDerivatives
from .sigmoid import SigmoidDerivatives
from .tanh import TanhDerivatives


derivatives_for = {
    Linear: LinearDerivatives,
    LinearConcat: LinearConcatDerivatives,
    Conv2d: Conv2DDerivatives,
    Conv2dConcat: Conv2DConcatDerivatives,
    AvgPool2d: AvgPool2DDerivatives,
    MaxPool2d: MaxPool2DDerivatives,
    ZeroPad2d: ZeroPad2dDerivatives,
    Dropout: DropoutDerivatives,
    ReLU: ReLUDerivatives,
    Tanh: TanhDerivatives,
    Sigmoid: SigmoidDerivatives
}
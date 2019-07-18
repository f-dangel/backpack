from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives

from .maxpool2d import MaxPool2DDerivatives
from .zeropad2d import ZeroPad2dDerivatives
from .dropout import DropoutDerivatives
from .relu import ReLUDerivatives
from .sigmoid import SigmoidDerivatives
from .tanh import TanhDerivatives
from .linear import LinearDerivatives

from torch.nn import Sigmoid, Tanh, ReLU, Dropout, ZeroPad2d, MaxPool2d, Linear, AvgPool2d

derivatives_for = {
    Linear: LinearDerivatives,
    AvgPool2d: AvgPool2DDerivatives,
    MaxPool2d: MaxPool2DDerivatives,
    ZeroPad2d: ZeroPad2dDerivatives,
    Dropout: DropoutDerivatives,
    ReLU: ReLUDerivatives,
    Tanh: TanhDerivatives,
    Sigmoid: SigmoidDerivatives
}
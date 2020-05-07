from torch.nn import (
    AvgPool2d,
    Conv2d,
    ConvTranspose2d,
    CrossEntropyLoss,
    Dropout,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from .avgpool2d import AvgPool2DDerivatives
from .conv2d import Conv2DDerivatives
from .conv_transpose2d import ConvTranspose2DDerivatives
from .crossentropyloss import CrossEntropyLossDerivatives
from .dropout import DropoutDerivatives
from .linear import LinearDerivatives
from .maxpool2d import MaxPool2DDerivatives
from .mseloss import MSELossDerivatives
from .relu import ReLUDerivatives
from .sigmoid import SigmoidDerivatives
from .tanh import TanhDerivatives
from .zeropad2d import ZeroPad2dDerivatives

derivatives_for = {
    Linear: LinearDerivatives,
    Conv2d: Conv2DDerivatives,
    AvgPool2d: AvgPool2DDerivatives,
    MaxPool2d: MaxPool2DDerivatives,
    ZeroPad2d: ZeroPad2dDerivatives,
    Dropout: DropoutDerivatives,
    ReLU: ReLUDerivatives,
    Tanh: TanhDerivatives,
    Sigmoid: SigmoidDerivatives,
    ConvTranspose2d: ConvTranspose2DDerivatives,
    CrossEntropyLoss: CrossEntropyLossDerivatives,
    MSELoss: MSELossDerivatives,
}

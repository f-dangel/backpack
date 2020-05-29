from torch.nn import (
    ELU,
    SELU,
    AvgPool2d,
    Conv2d,
    ConvTranspose2d,
    CrossEntropyLoss,
    Dropout,
    LeakyReLU,
    Linear,
    LogSigmoid,
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
from .elu import ELUDerivatives
from .leakyrelu import LeakyReLUDerivatives
from .linear import LinearDerivatives
from .logsigmoid import LogSigmoidDerivatives
from .maxpool2d import MaxPool2DDerivatives
from .mseloss import MSELossDerivatives
from .relu import ReLUDerivatives
from .selu import SELUDerivatives
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
    LeakyReLU: LeakyReLUDerivatives,
    LogSigmoid: LogSigmoidDerivatives,
    ELU: ELUDerivatives,
    SELU: SELUDerivatives,
    CrossEntropyLoss: CrossEntropyLossDerivatives,
    MSELoss: MSELossDerivatives,
}

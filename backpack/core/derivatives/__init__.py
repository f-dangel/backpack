from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    ELU,
    MSELoss,
    Dropout,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool2d,
    ReLU,
    SELU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from .avgpool2d import AvgPool2DDerivatives
from .conv2d import Conv2DDerivatives
from .crossentropyloss import CrossEntropyLossDerivatives
from .elu import ELUDerivatives
from .mseloss import MSELossDerivatives
from .dropout import DropoutDerivatives
from .leakyrelu import LeakyReLUDerivatives
from .linear import LinearDerivatives
from .logsigmoid import LogSigmoidDerivatives
from .maxpool2d import MaxPool2DDerivatives
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
    LeakyReLU: LeakyReLUDerivatives,
    LogSigmoid: LogSigmoidDerivatives,
    ELU: ELUDerivatives,
    SELU: SELUDerivatives,
    CrossEntropyLoss: CrossEntropyLossDerivatives,
    MSELoss: MSELossDerivatives,
}

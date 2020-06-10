from torch.nn import (
    AvgPool2d,
    Conv1d,
    Conv2d,
    Conv3d,
    CrossEntropyLoss,
    Dropout,
    ELU,
    Linear,
    LeakyReLU,
    LogSigmoid,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    SELU,
    Tanh,
    ZeroPad2d,
)

from .avgpool2d import AvgPool2DDerivatives
from .conv1d import Conv1DDerivatives
from .conv2d import Conv2DDerivatives
from .conv3d import Conv3DDerivatives
from .crossentropyloss import CrossEntropyLossDerivatives
from .elu import ELUDerivatives
from .dropout import DropoutDerivatives
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
    Conv1d: Conv1DDerivatives,
    Conv2d: Conv2DDerivatives,
    Conv3d: Conv3DDerivatives,
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

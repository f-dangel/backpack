from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from .avgpool1d import AvgPool1DDerivatives
from .avgpool2d import AvgPool2DDerivatives
from .avgpool3d import AvgPool3DDerivatives
from .conv1d import Conv1DDerivatives
from .conv2d import Conv2DDerivatives
from .conv3d import Conv3DDerivatives
from .conv_transpose1d import ConvTranspose1DDerivatives
from .conv_transpose2d import ConvTranspose2DDerivatives
from .conv_transpose3d import ConvTranspose3DDerivatives
from .crossentropyloss import CrossEntropyLossDerivatives
from .dropout import DropoutDerivatives
from .elu import ELUDerivatives
from .leakyrelu import LeakyReLUDerivatives
from .linear import LinearDerivatives
from .logsigmoid import LogSigmoidDerivatives
from .maxpool1d import MaxPool1DDerivatives
from .maxpool2d import MaxPool2DDerivatives
from .maxpool3d import MaxPool3DDerivatives
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
    AvgPool1d: AvgPool1DDerivatives,
    AvgPool2d: AvgPool2DDerivatives,
    AvgPool3d: AvgPool3DDerivatives,
    MaxPool1d: MaxPool1DDerivatives,
    MaxPool2d: MaxPool2DDerivatives,
    MaxPool3d: MaxPool3DDerivatives,
    ZeroPad2d: ZeroPad2dDerivatives,
    Dropout: DropoutDerivatives,
    ReLU: ReLUDerivatives,
    Tanh: TanhDerivatives,
    Sigmoid: SigmoidDerivatives,
    ConvTranspose1d: ConvTranspose1DDerivatives,
    ConvTranspose2d: ConvTranspose2DDerivatives,
    ConvTranspose3d: ConvTranspose3DDerivatives,
    LeakyReLU: LeakyReLUDerivatives,
    LogSigmoid: LogSigmoidDerivatives,
    ELU: ELUDerivatives,
    SELU: SELUDerivatives,
    CrossEntropyLoss: CrossEntropyLossDerivatives,
    MSELoss: MSELossDerivatives,
}

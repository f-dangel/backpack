"""Test functionality of `backpack.core.derivatives` module."""
from torch.nn import (
    ELU,
    LSTM,
    RNN,
    SELU,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Identity,
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

from backpack.core.derivatives.adaptive_avg_pool_nd import (
    AdaptiveAvgPool1dDerivatives,
    AdaptiveAvgPool2dDerivatives,
    AdaptiveAvgPool3dDerivatives,
)
from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.embedding import EmbeddingDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.lstm import LSTMDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.pad import PadDerivatives
from backpack.core.derivatives.permute import PermuteDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.slicing import SlicingDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.custom_module.branching import SumModule
from backpack.custom_module.pad import Pad
from backpack.custom_module.permute import Permute
from backpack.custom_module.scale_module import ScaleModule
from backpack.custom_module.slicing import Slicing

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
    RNN: RNNDerivatives,
    Permute: PermuteDerivatives,
    LSTM: LSTMDerivatives,
    AdaptiveAvgPool1d: AdaptiveAvgPool1dDerivatives,
    AdaptiveAvgPool2d: AdaptiveAvgPool2dDerivatives,
    AdaptiveAvgPool3d: AdaptiveAvgPool3dDerivatives,
    BatchNorm1d: BatchNormNdDerivatives,
    BatchNorm2d: BatchNormNdDerivatives,
    BatchNorm3d: BatchNormNdDerivatives,
    Embedding: EmbeddingDerivatives,
    ScaleModule: ScaleModuleDerivatives,
    Identity: ScaleModuleDerivatives,
    SumModule: SumModuleDerivatives,
    Pad: PadDerivatives,
    Slicing: SlicingDerivatives,
}

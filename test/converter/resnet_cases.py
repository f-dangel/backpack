"""Contains example ResNets to be used in tests."""
from torch import flatten, tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    MSELoss,
    ReLU,
    Sequential,
    Tanh,
)
from torchvision.models.resnet import BasicBlock, conv1x1


class ResNet1(Module):
    """Small ResNet."""

    def __init__(self, in_dim: int = 2, out_dim: int = 10):
        """Initialization.

        Args:
            in_dim: input dimensions
            out_dim: output dimensions
        """
        super().__init__()
        self.net = Sequential(
            Linear(in_dim, out_dim),
            Tanh(),
            Linear(out_dim, out_dim),
            Tanh(),
            Linear(out_dim, in_dim),
        )

    def forward(self, input):
        """Forward pass. One Euler step.

        Args:
            input: input tensor

        Returns:
            result
        """
        x = self.net(input)
        return input + x * 0.1

    input_test = tensor([[1.0, 2.0]])
    target_test = tensor([[1.0, 1.0]])
    loss_test = MSELoss()


class ResNet2(Module):
    """Replicates resnet18 but a lot smaller."""

    num_classes: int = 3
    batch_size: int = 2
    picture_width: int = 7
    inplanes = 2

    input_test = (batch_size, 3, picture_width, picture_width)
    target_test = (batch_size, num_classes)
    loss_test = MSELoss()

    def __init__(self):
        """Initialization."""
        super().__init__()
        self.inplanes = ResNet2.inplanes

        self.conv1 = Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, ResNet2.inplanes, 2)
        self.layer2 = self._make_layer(BasicBlock, 2 * ResNet2.inplanes, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4 * ResNet2.inplanes, 2, stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(4 * ResNet2.inplanes, self.num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            result
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        """Creates a concatenation of blocks in the ResNet.

        This function is similar to the one in torchvision/resnets.
        https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

        Args:
            block: basic block to use (with one skip connection)
            planes: number of parallel planes
            blocks: number of sequential blocks
            stride: factor between input and output planes

        Returns:
            a sequence of blocks
        """
        norm_layer = BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, 1, 64, 1, norm_layer)
        ]
        self.inplanes = planes * block.expansion
        layers += [
            block(
                self.inplanes,
                planes,
                groups=1,
                base_width=64,
                dilation=1,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        ]

        return Sequential(*layers)

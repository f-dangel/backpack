"""Test cases for the converter.

Network with resnet18
Network with inplace activation
Network with parameter-free module used in multiple places
Network with flatten operation
Network with multiply operation
Network with add operation
"""
import abc
from typing import List, Type

from torch import Tensor, flatten, rand
from torch.nn import Linear, Module, ReLU
from torchvision.models import resnet18, wide_resnet50_2


class ConverterModule(Module, abc.ABC):
    """Interface class for test modules for converter."""

    def __init__(self):  # noqa: D107
        super().__init__()

    @abc.abstractmethod
    def input_fn(self) -> Tensor:
        """Generate a fitting input for the module.

        Returns:
            an input
        """
        return


CONVERTER_MODULES: List[Type[ConverterModule]] = []


class _ResNet18(ConverterModule):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(num_classes=4).eval()

    def forward(self, x):
        return self.resnet18(x)

    def input_fn(self) -> Tensor:
        return rand(2, 3, 7, 7)


class _WideResNet50(ConverterModule):
    def __init__(self):
        super().__init__()
        self.wide_resnet50_2 = wide_resnet50_2(num_classes=4).eval()

    def forward(self, x):
        return self.wide_resnet50_2(x)

    def input_fn(self) -> Tensor:
        return rand(2, 3, 7, 7)


class _InplaceActivation(ConverterModule):
    def __init__(self):
        self.in_dim = 3
        out_dim = 2
        super().__init__()
        self.linear = Linear(self.in_dim, out_dim)
        self.relu = ReLU(inplace=True)
        self.linear2 = Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def input_fn(self) -> Tensor:
        return rand(3, self.in_dim)


class _MultipleUsages(ConverterModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 3
        out_dim = 2
        self.linear = Linear(self.in_dim, out_dim)
        self.relu = ReLU()
        self.linear2 = Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.relu(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

    def input_fn(self) -> Tensor:
        return rand(3, self.in_dim)


class _FlattenNetwork(ConverterModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 4
        out_dim = 3
        self.linear = Linear(self.in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = flatten(x, 1)
        return x

    def input_fn(self) -> Tensor:
        return rand(3, 2, 2, self.in_dim)


class _Multiply(ConverterModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 4
        out_dim = 3
        self.linear = Linear(self.in_dim, out_dim)

    def forward(self, x):
        x = x * 2.5
        x = self.linear(x)
        x = 0.5 * x
        return x

    def input_fn(self) -> Tensor:
        return rand(2, self.in_dim)


class _Add(ConverterModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 3
        out_dim = 2
        self.linear = Linear(self.in_dim, self.in_dim)
        self.linear1 = Linear(self.in_dim, out_dim)
        self.linear2 = Linear(self.in_dim, out_dim)
        self.relu = ReLU()

    def forward(self, x):
        x = self.linear(x)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = x1 + x2
        x = self.relu(x)
        return x

    def input_fn(self) -> Tensor:
        return rand(3, self.in_dim)


CONVERTER_MODULES += [
    _ResNet18,
    _WideResNet50,
    _InplaceActivation,
    _MultipleUsages,
    _FlattenNetwork,
    _Multiply,
    _Add,
]

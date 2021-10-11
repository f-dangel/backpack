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

from torch import Tensor, flatten, permute, rand, randint, transpose, zeros_like
from torch.nn import (
    LSTM,
    RNN,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Linear,
    Module,
    MSELoss,
    ReLU,
)
from torchvision.models import resnet18, wide_resnet50_2


class ConverterModule(Module, abc.ABC):
    """Interface class for test modules for converter."""

    @abc.abstractmethod
    def input_fn(self) -> Tensor:
        """Generate a fitting input for the module.

        Returns:
            an input
        """
        return

    def loss_fn(self) -> Module:
        """The loss function.

        Returns:
            loss function
        """
        return MSELoss()


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
        super().__init__()
        self.batch_size = 3
        self.in_dim = 3
        out_dim = 2
        self.linear = Linear(self.in_dim, out_dim)
        self.relu = ReLU(inplace=True)
        self.linear2 = Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, self.in_dim)


class _MultipleUsages(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 3
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
        return rand(self.batch_size, self.in_dim)


class _FlattenNetwork(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 3
        self.in_dim = (2, 2, 4)
        out_dim = 3
        self.linear = Linear(self.in_dim[2], out_dim)
        self.linear2 = Linear(self.in_dim[1] * out_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = flatten(x, 2)  # built-in function flatten
        x = self.linear2(x)
        x = x.flatten(1)  # method flatten
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, *self.in_dim)


class _Multiply(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.in_dim = 4
        out_dim = 3
        self.linear = Linear(self.in_dim, out_dim)

    def forward(self, x):
        x = x * 2.5  # built-in method multiply (Tensor-float)
        x = self.linear(x)
        x = 0.5 * x  # built-in method multiply (float-Tensor)
        x = x.multiply(3.1415)  # method multiply
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, self.in_dim)


class _Add(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 3
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
        x = x1 + x2  # built-in method add
        x = self.relu(x)
        x = x.add(x2)  # method add
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, self.in_dim)


class _Permute(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 3
        self.in_dim = (5, 3)
        out_dim = 2
        self.linear = Linear(self.in_dim[-1], out_dim)
        self.linear2 = Linear(self.in_dim[-2], out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = x.permute(0, 2, 1)  # method permute
        x = self.linear2(x)
        x = permute(x, (0, 2, 1))  # function permute
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, *self.in_dim)

    def loss_fn(self) -> Module:
        return CrossEntropyLoss()


class _Transpose(ConverterModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 3
        self.in_dim = (5, 3)
        out_dim = 2
        out_dim2 = 3
        self.linear = Linear(self.in_dim[-1], out_dim)
        self.linear2 = Linear(self.in_dim[-2], out_dim2)

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(1, 2)  # method transpose
        x = self.linear2(x)
        x = transpose(x, 1, 2)  # function transpose
        return x

    def input_fn(self) -> Tensor:
        return rand(self.batch_size, *self.in_dim)

    def loss_fn(self) -> Module:
        return CrossEntropyLoss()


class _TolstoiCharRNN(ConverterModule):
    def __init__(self):
        super(_TolstoiCharRNN, self).__init__()
        self.batch_size = 8
        self.hidden_dim = 64
        self.num_layers = 2
        self.seq_len = 15
        self.vocab_size = 25

        self.embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_dim
        )
        self.dropout = Dropout(p=0.2)
        self.lstm = LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.36,
            batch_first=True,
        )
        self.lstm.bias_ih_l0.data = zeros_like(self.lstm.bias_ih_l0)
        self.lstm.bias_ih_l1.data = zeros_like(self.lstm.bias_ih_l1)
        self.lstm.bias_ih_l0.requires_grad = False
        self.lstm.bias_ih_l1.requires_grad = False
        self.dense = Linear(in_features=self.hidden_dim, out_features=self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, new_state = self.lstm(x)
        x = self.dropout(x)
        output = self.dense(x)
        output = output.permute(0, 2, 1)
        return output

    def input_fn(self) -> Tensor:
        return randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def loss_fn(self) -> Module:
        return CrossEntropyLoss()


class _TolstoiRNNVersion(_TolstoiCharRNN):
    def __init__(self):
        super(_TolstoiRNNVersion, self).__init__()
        self.lstm = RNN(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.36,
            batch_first=True,
        )


CONVERTER_MODULES += [
    _ResNet18,
    _WideResNet50,
    _InplaceActivation,
    _MultipleUsages,
    _FlattenNetwork,
    _Multiply,
    _Add,
    _Permute,
    _Transpose,
    _TolstoiCharRNN,
    _TolstoiRNNVersion,
]

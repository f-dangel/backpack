import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter
from torch import flatten, cat, Tensor, empty


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return flatten(input, start_dim=1, end_dim=-1)


class SkipConnection(Module):
    pass


class LinearConcat(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        lin = Linear(in_features, out_features, bias=bias)

        if bias:
            self.weight = Parameter(empty(size=(out_features, in_features + 1)))
            self.weight.data = cat(
                [lin.weight.data, lin.bias.data.unsqueeze(-1)], dim=1
            )
        else:
            self.weight = Parameter(empty(size=(out_features, in_features)))
            self.weight.data = lin.weight.data

        self.input_features = in_features
        self.output_features = out_features
        self.bias = bias

    def forward(self, input):
        if self.bias:
            return F.linear(
                input,
                self.weight.narrow(1, 0, self.input_features),
                self.weight.narrow(1, self.input_features, 1).squeeze(-1)
            )
        else:
            return F.linear(input, self.weight, None)

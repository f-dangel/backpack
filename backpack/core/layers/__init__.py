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
            self.weight = Parameter(
                empty(size=(out_features, in_features + 1)))
            self.weight.data = cat(
                [lin.weight.data, lin.bias.data.unsqueeze(-1)], dim=1)
        else:
            self.weight = Parameter(empty(size=(out_features, in_features)))
            self.weight.data = lin.weight.data

        self.input_features = in_features
        self.output_features = out_features
        self.bias = bias

    def forward(self, input):
        return F.linear(input, self._slice_weight(), self._slice_bias())

    def has_bias(self):
        return self.bias is True

    def homogeneous_input(self):
        input = self.input0
        if self.has_bias():
            input = self.append_ones(input)
        return input

    @staticmethod
    def append_ones(input):
        batch = input.shape[0]
        ones = torch.ones(batch, 1, device=input.device)
        return torch.cat([input, ones], dim=1)

    def _slice_weight(self):
        return self.weight.narrow(1, 0, self.input_features)

    def _slice_bias(self):
        if not self.has_bias():
            return None
        else:
            return self.weight.narrow(1, self.input_features, 1).squeeze(-1)

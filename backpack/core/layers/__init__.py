import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, Conv2d
from torch import flatten, cat, Tensor, empty
from ...utils.conv import unfold_func


class Flatten(Module):
    """
    NN module version of torch.nn.functional.flatten
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return flatten(input, start_dim=1, end_dim=-1)


class SkipConnection(Module):
    pass


class LinearConcat(Module):
    """
    Drop-in replacement for torch.nn.Linear with only one parameter.
    """
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
        self.__bias = bias

    def forward(self, input):
        return F.linear(input, self._slice_weight(), self._slice_bias())

    def has_bias(self):
        return self.__bias is True

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


class Conv2dConcat(Module):
    """
    Drop-in replacement for torch.nn.Conv2d with only one parameter.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode="zeros"):
        assert padding_mode is "zeros"
        assert groups == 1

        super().__init__()

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

        self._KERNEL_SHAPE = conv.weight.shape

        kernel_mat_shape = [out_channels, conv.weight.numel() // out_channels]
        kernel_mat = conv.weight.data.view(kernel_mat_shape)

        if bias:
            kernel_mat_shape[1] += 1
            kernel_mat = cat([kernel_mat, conv.bias.data.unsqueeze(-1)], dim=1)

            self.weight = Parameter(empty(size=kernel_mat_shape))
            self.weight.data = kernel_mat

        else:
            self.weight = Parameter(empty(size=kernel_mat_shape))
            self.weight.data = kernel_mat

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.transposed = conv.transposed
        self.output_padding = conv.output_padding
        self.groups = conv.groups
        self.padding_mode = padding_mode
        self.__bias = bias

    def forward(self, input):
        return F.conv2d(
            input,
            self._slice_weight(),
            bias=self._slice_bias(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)

    def has_bias(self):
        return self.__bias is True

    def homogeneous_unfolded_input(self):
        unfolded_input = unfold_func(self)(self.input0)
        if self.has_bias():
            unfolded_input = self.append_ones(unfolded_input)
        return unfolded_input

    @staticmethod
    def append_ones(input):
        batch, _, cols = input.shape
        ones = torch.ones(batch, 1, cols, device=input.device)
        return torch.cat([input, ones], dim=1)

    def _slice_weight(self):
        return self.weight.narrow(1, 0,
                                  self.weight.size(1) - 1).view(
                                      self._KERNEL_SHAPE)

    def _slice_bias(self):
        if not self.has_bias():
            return None
        else:
            return self.weight.narrow(1,
                                      self.weight.size(1) - 1, 1).squeeze(-1)

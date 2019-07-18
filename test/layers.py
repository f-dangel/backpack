from torch import nn
from backpack.core.layers import LinearConcat, Conv2dConcat

LINEARS = {
    'Linear': nn.Linear,
    'LinearConcat': LinearConcat,
}

ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}

CONVS = {
    'Conv2d': nn.Conv2d,
    'Conv2dConcat': Conv2dConcat,
}

PADDINGS = {
    'ZeroPad2d': nn.ZeroPad2d,
}

POOLINGS = {
    'MaxPool2d': nn.MaxPool2d,
    'AvgPool2d': nn.AvgPool2d,
}

from torch import nn

LINEARS = {
    "Linear": nn.Linear,
}

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}

CONVS = {
    "Conv2d": nn.Conv2d,
}

PADDINGS = {
    "ZeroPad2d": nn.ZeroPad2d,
}

POOLINGS = {
    "MaxPool2d": nn.MaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
}

BN = {
    "BatchNorm1d": nn.BatchNorm1d,
}

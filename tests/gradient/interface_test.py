"""
Test of the interface - calls every method that needs implementation
"""

import torch
from bpexts.gradient.conv2d import Conv2d as Conv2d
from bpexts.gradient.linear import Linear as Linear
import bpexts.gradient.config as config


def dummy_forward_pass():
    N = 3
    CONV_IN = 3
    CONV_OUT = 2
    DIM_X = 4
    DIM_Y = 3
    S = 1
    K = 2
    D_OUT = 2
    P = 0

    input = torch.randn(N, CONV_IN, DIM_X, DIM_Y)

    conv = Conv2d(
        in_channels=CONV_IN, out_channels=CONV_OUT,
        kernel_size=(K, K), stride=(S, S), padding=(P, P), bias=True
    )

    act = torch.nn.functional.relu

    lin = Linear(
        in_features=CONV_OUT * (DIM_X - S) * (DIM_Y - S),
        out_features=D_OUT, bias=True
    )

    def forward():
        return torch.sum(lin(act(conv(input)).view(N, -1))**2)

    return forward, (conv.weight, lin.weight)


forward_func, weights = dummy_forward_pass()


FEATURES_TO_ATTRIBUTES = {
    config.GRAD: "grad",
    config.BATCH_GRAD: "grad_batch",
    config.SUM_GRAD_SQUARED: "sum_grad_squared",
    config.GRAD_VAR: "grad_var",
}


def interface_test(feature):
    with config.bpexts(feature):
        forward_func().backward()
    for w in weights:
        assert hasattr(w, FEATURES_TO_ATTRIBUTES[feature])


def test_interface_grad():
    interface_test(config.GRAD)


def test_interface_batch_grad():
    interface_test(config.BATCH_GRAD)


def test_interface_sum_grad_squared():
    interface_test(config.SUM_GRAD_SQUARED)


def test_interface_grad_var():
    interface_test(config.GRAD_VAR)

"""
Test of the interface - calls every method that needs implementation
"""

import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss
from bpexts.gradient import extend, bpexts
import bpexts.gradient.extensions as ext


def dummy_forward_pass():
    N = 3
    D_IN = 5
    D_H = 10
    D_OUT = 2

    X = torch.randn(N, D_IN)
    Y = torch.randint(2, size=(N,))

    lin1 = extend(Linear(in_features=D_IN, out_features=D_H, bias=True))
    act = extend(ReLU())
    lin2 = extend(Linear(in_features=D_H, out_features=D_OUT, bias=True))
    loss = extend(CrossEntropyLoss())

    def model(x):
        return lin2(act(lin1(x)))

    def forward():
        return loss(model(X), Y)

    return forward, (lin1.weight, lin1.bias, lin2.weight, lin2.bias)


forward_func, weights = dummy_forward_pass()


FEATURES_TO_ATTRIBUTES = {
    ext.GRAD: "grad",
    ext.BATCH_GRAD: "grad_batch",
    ext.SUM_GRAD_SQUARED: "sum_grad_squared",
    ext.DIAG_GGN: "diag_ggn",
}


def interface_test(feature):
    with bpexts(feature):
        forward_func().backward()
    for w in weights:
        assert hasattr(w, FEATURES_TO_ATTRIBUTES[feature])


def test_interface_grad():
    interface_test(ext.GRAD)


def test_interface_batch_grad():
    interface_test(ext.BATCH_GRAD)


def test_interface_sum_grad_squared():
    interface_test(ext.SUM_GRAD_SQUARED)


def test_interface_diag_ggn():
    interface_test(ext.DIAG_GGN)

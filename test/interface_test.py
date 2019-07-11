"""
Test of the interface - calls every method that needs implementation
"""

import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss
from torch.nn import Sequential
from torch.nn import Conv2d
from backpack.core.layers import Flatten
from backpack import extend, backpack
import backpack.extensions as ext


def dummy_forward_pass():
    N = 3
    D_IN = 5
    D_H = 10
    D_OUT = 2

    X = torch.randn(N, D_IN)
    Y = torch.randint(2, size=(N, ))

    lin1 = extend(Linear(in_features=D_IN, out_features=D_H, bias=True))
    act = extend(ReLU())
    lin2 = extend(Linear(in_features=D_H, out_features=D_OUT, bias=True))
    loss = extend(CrossEntropyLoss())

    def model(x):
        return lin2(act(lin1(x)))

    def forward():
        return loss(model(X), Y)

    return forward, (lin1.weight, lin2.weight), (lin1.bias, lin2.bias)


def dummy_forward_pass_conv():
    N, C, H, W = 2, 3, 4, 4
    X = torch.randn(N, C, H, W)
    Y = torch.randint(high=5, size=(N, ))
    conv = extend(Conv2d(3, 2, 2))
    lin = extend(Linear(18, 5))
    model = Sequential(conv, Flatten(), lin)
    loss = extend(CrossEntropyLoss())

    def forward():
        return loss(model(X), Y)

    return forward, (conv.weight, lin.weight), (conv.bias, lin.bias)


forward_func, weights, bias = dummy_forward_pass()
forward_func_conv, weights_conv, bias_conv = dummy_forward_pass_conv()

FEATURES_TO_ATTRIBUTES = {
    ext.GRAD: "grad",
    ext.BATCH_GRAD: "grad_batch",
    ext.SUM_GRAD_SQUARED: "sum_grad_squared",
    ext.DIAG_GGN: "diag_ggn",
    ext.KFLR: "kflr",
    ext.KFAC: "kfac",
    ext.CMP: "cmp",
}


def interface_test(feature,
                   weight_has_attr=True,
                   bias_has_attr=True,
                   use_conv=False):
    if use_conv:
        f, ws, bs = forward_func_conv, weights_conv, bias_conv
    else:
        f, ws, bs = forward_func, weights, bias

    with backpack(feature):
        f().backward()
    for w in ws:
        assert weight_has_attr == hasattr(w, FEATURES_TO_ATTRIBUTES[feature])
    for b in bs:
        assert bias_has_attr == hasattr(b, FEATURES_TO_ATTRIBUTES[feature])


def test_interface_grad():
    interface_test(ext.GRAD)


def test_interface_batch_grad():
    interface_test(ext.BATCH_GRAD)


def test_interface_sum_grad_squared():
    interface_test(ext.SUM_GRAD_SQUARED)


def test_interface_diag_ggn():
    interface_test(ext.DIAG_GGN)


def test_interface_kflr():
    interface_test(ext.KFLR, weight_has_attr=True, bias_has_attr=False)


def test_interface_kfac():
    interface_test(ext.KFAC, weight_has_attr=True, bias_has_attr=False)


def test_interface_cmp():
    interface_test(ext.CMP)


def test_interface_grad_conv():
    interface_test(ext.GRAD, use_conv=True)


def test_interface_batch_grad_conv():
    interface_test(ext.BATCH_GRAD, use_conv=True)


def test_interface_sum_grad_squared_conv():
    interface_test(ext.SUM_GRAD_SQUARED, use_conv=True)


def test_interface_diag_ggn_conv():
    interface_test(ext.DIAG_GGN, use_conv=True)


def test_interface_kflr_conv():
    interface_test(
        ext.KFLR, weight_has_attr=True, bias_has_attr=False, use_conv=True)


def test_interface_kfac_conv():
    interface_test(
        ext.KFAC, weight_has_attr=True, bias_has_attr=False, use_conv=True)


def test_interface_cmp_conv():
    interface_test(ext.CMP, use_conv=True)

"""
Test of the interface - calls every method that needs implementation
"""
import pytest
import torch
from torch.nn import Conv2d, CrossEntropyLoss, Linear, ReLU, Sequential

import backpack.extensions as new_ext
from backpack import backpack, extend


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

    return forward, (lin1.weight, lin2.weight), (lin1.bias, lin2.bias)


def dummy_forward_pass_conv():
    N, C, H, W = 2, 3, 4, 4
    X = torch.randn(N, C, H, W)
    Y = torch.randint(high=5, size=(N,))
    conv = Conv2d(3, 2, 2)
    lin = Linear(18, 5)
    model = extend(Sequential(conv, torch.nn.Flatten(), lin))
    loss = extend(CrossEntropyLoss())

    def forward():
        return loss(model(X), Y)

    return forward, (conv.weight, lin.weight), (conv.bias, lin.bias)


forward_func, weights, bias = dummy_forward_pass()
forward_func_conv, weights_conv, bias_conv = dummy_forward_pass_conv()


def interface_test(extension, weight_has_attr=True, bias_has_attr=True, use_conv=False):
    if use_conv:
        f, ws, bs = forward_func_conv, weights_conv, bias_conv
    else:
        f, ws, bs = forward_func, weights, bias

    with backpack(extension):
        f().backward()
    for w in ws:
        assert weight_has_attr == hasattr(w, extension.savefield)
    for b in bs:
        assert bias_has_attr == hasattr(b, extension.savefield)


def test_interface_batch_grad():
    interface_test(new_ext.BatchGrad())


def test_interface_variance():
    interface_test(new_ext.Variance())


def test_interface_sum_grad_squared():
    interface_test(new_ext.SumGradSquared())


def test_interface_diag_ggn():
    interface_test(new_ext.DiagGGN())


def test_interface_diag_h():
    interface_test(new_ext.DiagHessian())


def test_interface_kflr():
    interface_test(new_ext.KFLR())


def test_interface_kfra():
    interface_test(new_ext.KFRA())


def test_interface_kfac():
    interface_test(new_ext.KFAC())


def test_interface_hmp():
    interface_test(new_ext.HMP())


def test_interface_ggnmp():
    interface_test(new_ext.PCHMP())


def test_interface_pchmp():
    interface_test(new_ext.GGNMP())


@pytest.mark.skip()
def test_interface_hbp():
    interface_test(new_ext.HBP())


def test_interface_batch_grad_conv():
    interface_test(new_ext.BatchGrad(), use_conv=True)


def test_interface_sum_grad_squared_conv():
    interface_test(new_ext.SumGradSquared(), use_conv=True)


def test_interface_diag_ggn_conv():
    interface_test(new_ext.DiagGGN(), use_conv=True)


def test_interface_kflr_conv():
    interface_test(new_ext.KFLR(), use_conv=True)


def test_interface_kfra_conv():
    interface_test(new_ext.KFRA(), use_conv=True)


def test_interface_kfac_conv():
    interface_test(new_ext.KFAC(), use_conv=True)


@pytest.mark.skip()
def test_interface_hbp_conv():
    interface_test(new_ext.HBP(), use_conv=True)


def test_interface_hmp_conv():
    interface_test(new_ext.HMP(), use_conv=True)


def test_interface_ggnmp_conv():
    interface_test(new_ext.PCHMP(), use_conv=True)


def test_interface_pchmp_conv():
    interface_test(new_ext.GGNMP(), use_conv=True)

import pytest
import torch
from torch.nn import Linear, Conv2d

from backpack import extend
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product


################################################################################
# Data and helpers
################################################################################

ATOL = 1e-3
RTOL = 1e-3


def data():
    N, Cin, Hin, Win = 10, 10, 32, 32
    Cout, KernelH, KernelW = 25, 5, 5

    X = torch.randn(N, Cin, Hin, Win, requires_grad=True)
    module = extend(Conv2d(Cin, Cout, (KernelH, KernelW)))
    out = module(X)

    Hout = Hin - (KernelH - 1)
    Wout = Win - (KernelW - 1)
    v = torch.randn(N, Cout, Hout, Wout)

    return X, module , out, v


def ag_jtv_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, X, v, detach=False
    )[0]


def bp_jtv_func(X, module, out, v):
    return lambda: Conv2DDerivatives().jac_t_mat_prod(
        module, None, None, v.view(v.shape[0], -1).unsqueeze(-1)
    ).squeeze(2)


def ag_jtv_weight_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, module.weight, v, detach=False
    )[0]


def bp_jtv_weight_func(X, module, out, v):
    return lambda: Conv2DDerivatives().weight_jac_t_mat_prod(
        module, None, None, v.view(v.shape[0], -1).unsqueeze(-1)
    )


def ag_jtv_bias_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, module.bias, v, detach=False
    )[0]


def bp_jtv_bias_func(X, module, out, v):
    return lambda: Conv2DDerivatives().bias_jac_t_mat_prod(
        module, None, None, v.view(v.shape[0], -1).unsqueeze(-1)
    )


################################################################################
# Correctness tests
################################################################################


def test_jtv_ag_vs_bp():
    X, module, out, v = data()
    A = ag_jtv_func(X, module, out, v)()
    B = bp_jtv_func(X, module, out, v)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


def test_jtv_weight_ag_vs_bp():
    X, module, out, v = data()
    A = ag_jtv_weight_func(X, module, out, v)()
    B = bp_jtv_weight_func(X, module, out, v)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


def test_jtv_bias_ag_vs_bp():
    X, module, out, v = data()
    A = ag_jtv_bias_func(X, module, out, v)()
    B = bp_jtv_bias_func(X, module, out, v)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


################################################################################
# Benchmarks
################################################################################


def test_jtv_conv2d_ag(benchmark):
    X, module, out, v = data()
    benchmark(ag_jtv_func(X, module, out, v))


def test_jtv_conv2d_bp(benchmark):
    X, module, out, v = data()
    benchmark(bp_jtv_func(X, module, out, v))


def test_jtv_conv2d_weight_ag(benchmark):
    X, module, out, v = data()
    benchmark(ag_jtv_weight_func(X, module, out, v))


def test_jtv_conv2d_weight_bp(benchmark):
    X, module, out, v = data()
    benchmark(bp_jtv_weight_func(X, module, out, v))


def test_jtv_conv2d_bias_ag(benchmark):
    X, module, out, v = data()
    benchmark(ag_jtv_bias_func(X, module, out, v))


def test_jtv_conv2d_bias_bp(benchmark):
    X, module, out, v = data()
    benchmark(bp_jtv_bias_func(X, module, out, v))

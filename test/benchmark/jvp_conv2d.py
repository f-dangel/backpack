import torch
from torch.nn import Conv2d

from backpack import extend
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product

################################################################################
# Data and helpers
################################################################################
from backpack.hessianfree.rop import jacobian_vector_product

ATOL = 1e-3
RTOL = 1e-3


def data():
    N, Cin, Hin, Win = 100, 10, 32, 32
    Cout, KernelH, KernelW = 25, 5, 5

    X = torch.randn(N, Cin, Hin, Win, requires_grad=True)
    module = extend(Conv2d(Cin, Cout, (KernelH, KernelW)))
    out = module(X)

    Hout = Hin - (KernelH - 1)
    Wout = Win - (KernelW - 1)
    vout = torch.randn(N, Cout, Hout, Wout)
    vin = torch.randn(N, Cin, Hin, Win)

    return X, module, out, vout, vin


def ag_jtv_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, X, vin, detach=False
    )[0].contiguous()


def ag_jv_func(X, module, out, vin, vout):
    return lambda: jacobian_vector_product(
        out, X, vout, detach=False
    )[0].contiguous()


def bp_jtv_func(X, module, out, vin, vout):
    return lambda: Conv2DDerivatives().jac_t_mat_prod(
        module, None, None, vin.view(vin.shape[0], -1).unsqueeze(-1)
    ).squeeze(2).contiguous()


def bp_jv_func(X, module, out, vin, vout):
    return lambda: Conv2DDerivatives().jac_mat_prod(
        module, None, None, vout.view(vout.shape[0], -1).unsqueeze(-1)
    ).squeeze(2).contiguous()


def ag_jtv_weight_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, module.weight, vin, detach=False
    )[0]


def bp_jtv_weight_func(X, module, out, vin, vout):
    return lambda: Conv2DDerivatives().weight_jac_t_mat_prod(
        module, None, None, vin.view(vin.shape[0], -1).unsqueeze(-1)
    )


def ag_jtv_bias_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, module.bias, vin, detach=False
    )[0]


def bp_jtv_bias_func(X, module, out, vin, vout):
    return lambda: Conv2DDerivatives().bias_jac_t_mat_prod(
        module, None, None, vin.view(vin.shape[0], -1).unsqueeze(-1)
    )


################################################################################
# Correctness tests
################################################################################


def test_jtv_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_func(X, module, out, vin, vout)()
    B = bp_jtv_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


def test_jv_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jv_func(X, module, out, vin, vout)()
    B = bp_jv_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


def test_jtv_weight_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_weight_func(X, module, out, vin, vout)()
    B = bp_jtv_weight_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


def test_jtv_bias_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_bias_func(X, module, out, vin, vout)()
    B = bp_jtv_bias_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


################################################################################
# Benchmarks
################################################################################


def test_jtv_conv2d_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_func(X, module, out, vin, vout))


def test_jtv_conv2d_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_func(X, module, out, vin, vout))


def test_jv_conv2d_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jv_func(X, module, out, vin, vout))


def test_jv_conv2d_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jv_func(X, module, out, vin, vout))


def test_jtv_conv2d_weight_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_weight_func(X, module, out, vin, vout))


def test_jtv_conv2d_weight_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_weight_func(X, module, out, vin, vout))


def test_jtv_conv2d_bias_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_bias_func(X, module, out, vin, vout))


def test_jtv_conv2d_bias_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_bias_func(X, module, out, vin, vout))

import torch
from torch.nn import MaxPool2d

from backpack import extend
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product

ATOL = 1e-5
RTOL = 1e-5


################################################################################
# Data and helpers
################################################################################


def data():
    N, C, Hin, Win = 100, 10, 32, 32
    KernelSize = 4
    Hout = int(Hin / KernelSize)
    Wout = int(Win / KernelSize)

    X = torch.randn(N, C, Hin, Win, requires_grad=True)
    module = extend(MaxPool2d(KernelSize))
    out = module(X)

    vin = torch.randn(N, C, Hin, Win)
    vout = torch.randn(N, C, Hout, Wout)

    return X, module, out, vout, vin


def ag_jtv_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, X, vin, detach=False
    )[0].contiguous()


def bp_jtv_func(X, module, out, vin, vout):
    return lambda: MaxPool2DDerivatives().jac_t_mat_prod(
        module, None, None, vin.view(vin.shape[0], -1).unsqueeze(-1)
    ).squeeze(2).contiguous()


def ag_jv_func(X, module, out, vin, vout):
    return lambda: jacobian_vector_product(
        out, X, vout, detach=False
    )[0].contiguous()


def bp_jv_func(X, module, out, vin, vout):
    return lambda: MaxPool2DDerivatives().jac_mat_prod(
        module, None, None, vout.view(vout.shape[0], -1).unsqueeze(-1)
    ).squeeze(2).contiguous()


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


################################################################################
# Benchmarks
################################################################################


def test_jtv_maxpool2d_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_func(X, module, out, vin, vout))


def test_jtv_maxpool2d_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_func(X, module, out, vin, vout))


def test_jv_maxpool2d_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jv_func(X, module, out, vin, vout))


def test_jv_maxpool2d_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jv_func(X, module, out, vin, vout))

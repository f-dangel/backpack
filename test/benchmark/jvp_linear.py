import pytest
import torch
from torch.nn import Linear

from backpack import extend
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product

################################################################################
# Data and helpers
################################################################################
from backpack.hessianfree.rop import jacobian_vector_product


def data():
    N, D1, D2 = 10, 10, 10

    X = torch.randn(N, D1, requires_grad=True)
    linear = extend(Linear(D1, D2))
    out = linear(X)

    vin = torch.randn(N, D2)
    vout = torch.randn(N, D1)

    return X, linear, out, vin, vout


def ag_jtv_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, X, vin, detach=False
    )[0]


def ag_jv_func(X, module, out, vin, vout):
    return lambda: jacobian_vector_product(
        out, X, vout, detach=False
    )[0]


def bp_jtv_func(X, module, out, vin, vout):
    return lambda: LinearDerivatives().jac_t_mat_prod(
        module, None, None, vin.unsqueeze(2)
    ).squeeze(2)


def bp_jv_func(X, module, out, vin, vout):
    return lambda: LinearDerivatives().jac_mat_prod(
        module, None, None, vout.unsqueeze(2)
    ).squeeze(2)


def ag_jtv_weight_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, module.weight, vin, detach=False
    )[0]


def bp_jtv_weight_func(X, module, out, vin, vout):
    return lambda: LinearDerivatives().weight_jac_t_mat_prod(
        module, None, None, vin.unsqueeze(2)
    )


def ag_jtv_bias_func(X, module, out, vin, vout):
    return lambda: transposed_jacobian_vector_product(
        out, module.bias, vin, detach=False
    )[0]


def bp_jtv_bias_func(X, module, out, vin, vout):
    return lambda: LinearDerivatives().bias_jac_t_mat_prod(
        module, None, None, vin.unsqueeze(2)
    )


################################################################################
# Correctness tests
################################################################################


def test_jtv_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_func(X, module, out, vin, vout)()
    B = bp_jtv_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B)


def test_jv_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_func(X, module, out, vin, vout)()
    B = bp_jtv_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B)


def test_jtv_weight_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_weight_func(X, module, out, vin, vout)()
    B = bp_jtv_weight_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A))


def test_jtv_bias_ag_vs_bp():
    X, module, out, vin, vout = data()
    A = ag_jtv_bias_func(X, module, out, vin, vout)()
    B = bp_jtv_bias_func(X, module, out, vin, vout)()
    assert torch.allclose(A, B.view_as(A))


################################################################################
# Benchmarks
################################################################################


def test_jtv_linear_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_func(X, module, out, vin, vout))


def test_jv_linear_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jv_func(X, module, out, vin, vout))


def test_jtv_linear_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_func(X, module, out, vin, vout))


def test_jv_linear_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jv_func(X, module, out, vin, vout))


def test_jtv_linear_weight_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_weight_func(X, module, out, vin, vout))


def test_jtv_linear_weight_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_weight_func(X, module, out, vin, vout))


def test_jtv_linear_bias_ag(benchmark):
    X, module, out, vin, vout = data()
    benchmark(ag_jtv_bias_func(X, module, out, vin, vout))


def test_jtv_linear_bias_bp(benchmark):
    X, module, out, vin, vout = data()
    benchmark(bp_jtv_bias_func(X, module, out, vin, vout))

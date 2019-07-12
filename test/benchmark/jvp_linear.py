import pytest
import torch
from torch.nn import Linear

from backpack import extend
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product


################################################################################
# Data and helpers
################################################################################


def data():
    N, D1, D2 = 10, 10, 10

    X = torch.randn(N, D1, requires_grad=True)
    linear = extend(Linear(D1, D2))
    out = linear(X)

    v = torch.randn(N, D2)

    return X, linear, out, v


def ag_jtv_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, X, v, detach=False
    )[0]


def bp_jtv_func(X, module, out, v):
    return lambda: LinearDerivatives().jac_t_mat_prod(
        module, None, None, v.unsqueeze(2)
    ).squeeze(2)


def ag_jtv_weight_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, module.weight, v, detach=False
    )[0]


def bp_jtv_weight_func(X, module, out, v):
    return lambda: LinearDerivatives().weight_jac_t_mat_prod(
        module, None, None, v.unsqueeze(2)
    )


def ag_jtv_bias_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, module.bias, v, detach=False
    )[0]


def bp_jtv_bias_func(X, module, out, v):
    return lambda: LinearDerivatives().bias_jac_t_mat_prod(
        module, None, None, v.unsqueeze(2)
    )


################################################################################
# Correctness tests
################################################################################


def test_jtv_ag_vs_bp():
    X, linear, out, v = data()
    A = ag_jtv_func(X, linear, out, v)()
    B = bp_jtv_func(X, linear, out, v)()
    assert torch.allclose(A, B)


def test_jtv_weight_ag_vs_bp():
    X, linear, out, v = data()
    A = ag_jtv_weight_func(X, linear, out, v)()
    B = bp_jtv_weight_func(X, linear, out, v)()
    assert torch.allclose(A, B.view_as(A))


def test_jtv_bias_ag_vs_bp():
    X, linear, out, v = data()
    A = ag_jtv_bias_func(X, linear, out, v)()
    B = bp_jtv_bias_func(X, linear, out, v)()
    assert torch.allclose(A, B.view_as(A))


################################################################################
# Benchmarks
################################################################################


def test_jtv_linear_ag(benchmark):
    X, linear, out, v = data()
    benchmark(ag_jtv_func(X, linear, out, v))


def test_jtv_linear_bp(benchmark):
    X, linear, out, v = data()
    benchmark(bp_jtv_func(X, linear, out, v))


def test_jtv_linear_weight_ag(benchmark):
    X, linear, out, v = data()
    benchmark(ag_jtv_weight_func(X, linear, out, v))


def test_jtv_linear_weight_bp(benchmark):
    X, linear, out, v = data()
    benchmark(bp_jtv_weight_func(X, linear, out, v))


def test_jtv_linear_bias_ag(benchmark):
    X, linear, out, v = data()
    benchmark(ag_jtv_bias_func(X, linear, out, v))


def test_jtv_linear_bias_bp(benchmark):
    X, linear, out, v = data()
    benchmark(bp_jtv_bias_func(X, linear, out, v))

import pytest
import torch
from torch.nn import Dropout, ReLU, Tanh, Sigmoid

from backpack import extend
from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product

################################################################################
# Data and helpers
################################################################################

MODULES = [Dropout, ReLU, Tanh, Sigmoid]

derivatives_for = {
    Dropout: DropoutDerivatives,
    ReLU: ReLUDerivatives,
    Tanh: TanhDerivatives,
    Sigmoid: SigmoidDerivatives
}


def data(module_class):
    N, D = 100, 200

    X = torch.randn(N, D, requires_grad=True)
    module = extend(module_class())
    out = module(X)

    v = torch.randn(N, D)

    return X, module, out, v


def ag_jtv_func(X, module, out, v):
    return lambda: transposed_jacobian_vector_product(
        out, X, v, detach=False
    )[0]


def ag_jv_func(X, module, out, v):
    return lambda: jacobian_vector_product(
        out, X, v, detach=False
    )[0]


def bp_jtv_func(X, module, out, v):
    return lambda: derivatives_for[module.__class__]().jac_t_mat_prod(
        module, None, None, v.unsqueeze(2)
    ).squeeze(2)


def bp_jv_func(X, module, out, v):
    return lambda: derivatives_for[module.__class__]().jac_mat_prod(
        module, None, None, v.unsqueeze(2)
    ).squeeze(2)


################################################################################
# Correctness tests
################################################################################


@pytest.mark.parametrize("module_class", MODULES)
def test_jtv_ag_vs_bp(module_class):
    X, module, out, v = data(Dropout)
    A = ag_jtv_func(X, module, out, v)()
    B = bp_jtv_func(X, module, out, v)()
    assert torch.allclose(A, B)


@pytest.mark.parametrize("module_class", MODULES)
def test_jv_ag_vs_bp(module_class):
    X, module, out, v = data(Dropout)
    A = ag_jv_func(X, module, out, v)()
    B = bp_jv_func(X, module, out, v)()
    assert torch.allclose(A, B)


################################################################################
# Benchmarks
################################################################################


@pytest.mark.parametrize("module_class", MODULES)
def test_jtv_ag(module_class, benchmark):
    X, module, out, v = data(module_class)
    benchmark(ag_jtv_func(X, module, out, v))


@pytest.mark.parametrize("module_class", MODULES)
def test_jv_ag(module_class, benchmark):
    X, module, out, v = data(module_class)
    benchmark(ag_jv_func(X, module, out, v))


@pytest.mark.parametrize("module_class", MODULES)
def test_jtv_bp(module_class, benchmark):
    X, module, out, v = data(module_class)
    benchmark(bp_jtv_func(X, module, out, v))


@pytest.mark.parametrize("module_class", MODULES)
def test_jv_bp(module_class, benchmark):
    X, module, out, v = data(module_class)
    benchmark(bp_jv_func(X, module, out, v))

from functools import partial

import pytest
import torch
from backpack.core.derivatives import derivatives_for
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product
from torch import allclose

from . import jvp_data

ATOL = 1e-3
RTOL = 1e-3

PROBLEMS_DEF = {
    "Linear": jvp_data.data_linear,
    "LinearConcat": jvp_data.data_linearconcat,
    "Conv2d": jvp_data.data_conv2d,
    "Conv2dConcat": jvp_data.data_conv2dconcat,
    "AvgPool2d": jvp_data.data_avgpool2d,
    "MaxPool2d": jvp_data.data_maxpool2d,
    "ZeroPad2d": jvp_data.data_zeropad2d,
    "Dropout": jvp_data.data_dropout,
    "ReLU": jvp_data.data_relu,
    "Tanh": jvp_data.data_tanh,
    "Sigmoid": jvp_data.data_sigmoid,
}

DEVICES = {"cpu": "cpu"}
if torch.cuda.is_available():
    DEVICES["gpu"] = "cuda:0"

PROBLEMS_DATA = []
PROBLEMS_NAME = []
for devname, dev in DEVICES.items():
    for name, problemfunc in PROBLEMS_DEF.items():
        PROBLEMS_NAME.append(name + ":" + devname)
        PROBLEMS_DATA.append(partial(problemfunc, device=dev))


################################################################################
# Helpers
################################################################################


def skip_if_attribute_does_not_exists(module, attr):
    if getattr(module, attr, None) is None:
        pytest.skip("Attribute {} does not exists for {}".format(attr, module))


def contiguous_and_synchronized(v):
    if v.is_cuda:
        torch.cuda.synchronize()
    return v.contiguous()


################################################################################
# Jacobian-vector products definitions
################################################################################


def ag_jtv_func(X, out, vin):
    def f():
        r = transposed_jacobian_vector_product(out, X, vin)[0]
        return contiguous_and_synchronized(r)

    return f


def ag_jv_func(X, out, vout):
    def f():
        r = jacobian_vector_product(out, X, vout)[0]
        return contiguous_and_synchronized(r)

    return f


def bp_jtv_func(module, vin):
    def f():
        dx = derivatives_for[module.__class__]()
        r = dx.jac_t_mat_prod(module, None, None, vin)
        return contiguous_and_synchronized(r)

    return f


def bp_jv_func(module, vout):
    def f():
        dx = derivatives_for[module.__class__]()
        r = dx.jac_mat_prod(module, None, None, vout)
        return contiguous_and_synchronized(r)

    return f


def ag_jtv_weight_func(module, out, vin):
    def f():
        r = transposed_jacobian_vector_product(out, module.weight, vin)[0]
        return contiguous_and_synchronized(r)

    return f


def bp_jtv_weight_func(module, vin):
    def f():
        dx = derivatives_for[module.__class__]()
        r = dx.weight_jac_t_mat_prod(module, None, None, vin)
        return contiguous_and_synchronized(r)

    return f


def ag_jtv_bias_func(module, out, vin):
    def f():
        r = transposed_jacobian_vector_product(out, module.bias, vin)[0]
        return contiguous_and_synchronized(r)

    return f


def bp_jtv_bias_func(module, vin):
    def f():
        dx = derivatives_for[module.__class__]()
        r = dx.bias_jac_t_mat_prod(module, None, None, vin.unsqueeze(2))
        return contiguous_and_synchronized(r)

    return f


################################################################################
# Correctness tests
################################################################################


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_ag_vs_bp(func):
    data = func()
    A = ag_jtv_func(data["X"], data["out"], data["vin_ag"])()
    data = func()
    B = bp_jtv_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jv_ag_vs_bp(func):
    data = func()
    A = ag_jv_func(data["X"], data["out"], data["vout_ag"])()
    data = func()
    B = bp_jv_func(data["module"], data["vout_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_weight_ag_vs_bp(func):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "weight")
    A = ag_jtv_weight_func(data["module"], data["out"], data["vin_ag"])()
    data = func()
    B = bp_jtv_weight_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_bias_ag_vs_bp(func):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "bias")
    A = ag_jtv_bias_func(data["module"], data["out"], data["vin_ag"])()
    data = func()
    B = bp_jtv_bias_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


################################################################################
# Benchmarks
################################################################################


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_input_ag(func, benchmark):
    data = func()
    benchmark(ag_jtv_func(data["X"], data["out"], data["vin_ag"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jv_input_ag(func, benchmark):
    data = func()
    benchmark(ag_jv_func(data["X"], data["out"], data["vout_ag"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_input_bp(func, benchmark):
    data = func()
    benchmark(bp_jtv_func(data["module"], data["vin_bp"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jv_input_bp(func, benchmark):
    data = func()
    benchmark(bp_jv_func(data["module"], data["vout_bp"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_weight_ag(func, benchmark):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "weight")
    benchmark(ag_jtv_weight_func(data["module"], data["out"], data["vin_ag"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_weight_bp(func, benchmark):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "weight")
    benchmark(bp_jtv_weight_func(data["module"], data["vin_bp"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_bias_ag(func, benchmark):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "bias")
    benchmark(ag_jtv_bias_func(data["module"], data["out"], data["vin_ag"]))


@pytest.mark.parametrize("func", PROBLEMS_DATA, ids=PROBLEMS_NAME)
def test_jtv_bias_bp(func, benchmark):
    data = func()
    skip_if_attribute_does_not_exists(data["module"], "bias")
    benchmark(bp_jtv_bias_func(data["module"], data["vin_bp"]))

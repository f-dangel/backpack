import pytest
from backpack.core.derivatives import derivatives_for
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product
from torch import allclose

from .jvp_linear import data as data_linear
from .jvp_conv2d import data as data_conv2d
from .jvp_avgpool2d import data as data_avgpool2d

ATOL = 1e-3
RTOL = 1e-3

PROBLEMS = {
    "Linear": data_linear(),
    "Conv2d": data_conv2d(),
    "AvgPool2d": data_avgpool2d(),
}

PROBLEM_DATA = []
PROBLEM_NAME = []
for name, problem in PROBLEMS.items():
    PROBLEM_NAME.append(name)
    PROBLEM_DATA.append(problem)


################################################################################
# Helpers
################################################################################


def skip_if_attribute_does_not_exists(module, attr):
    if getattr(module, attr, None) is None:
        pytest.skip("Attribute {} does not exists for {}".format(attr, module))


################################################################################
# Jacobian-vector products definitions
################################################################################


def ag_jtv_func(X, out, vin):
    return lambda: transposed_jacobian_vector_product(
        out, X, vin, detach=False
    )[0].contiguous()


def ag_jv_func(X, out, vout):
    return lambda: jacobian_vector_product(
        out, X, vout, detach=False
    )[0].contiguous()


def bp_jtv_func(module, vin):
    return lambda: derivatives_for[module.__class__]().jac_t_mat_prod(
        module, None, None, vin
    ).squeeze(2).contiguous()


def bp_jv_func(module, vout):
    return lambda: derivatives_for[module.__class__]().jac_mat_prod(
        module, None, None, vout
    ).squeeze(2).contiguous()


def ag_jtv_weight_func(module, out, vin):
    skip_if_attribute_does_not_exists(module, "weight")
    return lambda: transposed_jacobian_vector_product(
        out, module.weight, vin, detach=False
    )[0].contiguous()


def bp_jtv_weight_func(module, vin):
    skip_if_attribute_does_not_exists(module, "weight")
    return lambda: derivatives_for[module.__class__]().weight_jac_t_mat_prod(
        module, None, None, vin
    ).contiguous()


def ag_jtv_bias_func(module, out, vin):
    skip_if_attribute_does_not_exists(module, "bias")
    return lambda: transposed_jacobian_vector_product(
        out, module.bias, vin, detach=False
    )[0].contiguous()


def bp_jtv_bias_func(module, vin):
    skip_if_attribute_does_not_exists(module, "bias")
    return lambda: derivatives_for[module.__class__]().bias_jac_t_mat_prod(
        module, None, None, vin.unsqueeze(2)
    ).contiguous()


################################################################################
# Correctness tests
################################################################################


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_ag_vs_bp(data):
    print(data["vin_bp"].shape)
    A = ag_jtv_func(data["X"], data["output"], data["vin_ag"])()
    B = bp_jtv_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jv_ag_vs_bp(data):
    A = ag_jv_func(data["X"], data["output"], data["vout_ag"])()
    B = bp_jv_func(data["module"], data["vout_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_weight_ag_vs_bp(data):
    skip_if_attribute_does_not_exists(data["module"], "weight")
    A = ag_jtv_weight_func(data["module"], data["output"], data["vin_ag"])()
    B = bp_jtv_weight_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_bias_ag_vs_bp(data):
    skip_if_attribute_does_not_exists(data["module"], "bias")
    A = ag_jtv_bias_func(data["module"], data["output"], data["vin_ag"])()
    B = bp_jtv_bias_func(data["module"], data["vin_bp"])()
    assert allclose(A, B.view_as(A), atol=ATOL, rtol=RTOL)


################################################################################
# Benchmarks
################################################################################


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_ag(data, benchmark):
    benchmark(ag_jtv_func(data["X"], data["output"], data["vin_ag"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jv_ag(data, benchmark):
    benchmark(ag_jv_func(data["X"], data["output"], data["vout_ag"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_bp(data, benchmark):
    benchmark(bp_jtv_func(data["module"], data["vin_bp"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jv_bp(data, benchmark):
    benchmark(bp_jv_func(data["module"], data["vout_bp"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_weight_ag(data, benchmark):
    skip_if_attribute_does_not_exists(data["module"], "weight")
    benchmark(ag_jtv_weight_func(data["module"], data["output"], data["vin_ag"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_weight_bp(data, benchmark):
    skip_if_attribute_does_not_exists(data["module"], "weight")
    benchmark(bp_jtv_weight_func(data["module"], data["vin_bp"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_bias_ag(data, benchmark):
    skip_if_attribute_does_not_exists(data["module"], "bias")
    benchmark(ag_jtv_bias_func(data["module"], data["output"], data["vin_ag"]))


@pytest.mark.parametrize("data", PROBLEM_DATA, ids=PROBLEM_NAME)
def test_jtv_bias_bp(data, benchmark):
    skip_if_attribute_does_not_exists(data["module"], "bias")
    benchmark(bp_jtv_bias_func(data["module"], data["vin_bp"]))

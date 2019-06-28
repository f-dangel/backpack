"""Testing class for comparison of first-order information and brute-force
 auto-differentiation."""

import torch
import numpy as np
import pytest
from .test_problems_convolutions import TEST_PROBLEMS as CONV_TEST_PROBLEMS
from .test_problems_linear import TEST_PROBLEMS as LIN_TEST_PROBLEMS
from .test_problems_activations import TEST_PROBLEMS as ACT_TEST_PROBLEMS
from .test_problems_pooling import TEST_PROBLEMS as POOL_TEST_PROBLEMS

if torch.cuda.is_available():
    DEVICES = {
        "cpu": "cpu",
        "gpu": "cuda:0",
    }
else:
    DEVICES = {
        "cpu": "cpu",
    }
TEST_PROBLEMS = {
    **CONV_TEST_PROBLEMS,
    **LIN_TEST_PROBLEMS,
    **ACT_TEST_PROBLEMS,
    **POOL_TEST_PROBLEMS,
}

ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in TEST_PROBLEMS.items():
        ALL_CONFIGURATIONS.append(tuple([prob, dev]))
        CONFIGURATION_IDS.append(probname + "-" + dev_name)


atol = 1e-5
rtol = 1e-5

###
# Helpers
###


def report_nonclose_values(x, y):
    x_numpy = x.data.cpu().numpy().flatten()
    y_numpy = y.data.cpu().numpy().flatten()

    close = np.isclose(
        x_numpy, y_numpy, atol=atol, rtol=rtol)
    where_not_close = np.argwhere(np.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print('{} versus {}. Ratio of {}'.format(x, y, y / x))


def check_sizes(*plists):
    for i in range(len(plists) - 1):
        assert len(plists[i]) == len(plists[i + 1])

    for params in zip(*plists):
        for i in range(len(params) - 1):
            assert params[i].size() == params[i + 1].size()


###
# Tests
###

@pytest.mark.parametrize("problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients(problem, device):
    problem.to(device)
    autograd_res = problem.batch_gradients_autograd()
    bpexts_res = problem.batch_gradients_bpexts()
    model = problem.model

    check_sizes(autograd_res, bpexts_res)

    for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
        report_nonclose_values(g1, g2)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients_sum_to_grad(problem, device):
    problem.to(device)
    model = problem.model
    autograd_res = problem.gradient_autograd()
    bpexts_batch_res = problem.batch_gradients_bpexts()
    bpexts_res = list([g.sum(0) for g in bpexts_batch_res])

    check_sizes(autograd_res, bpexts_res, list(model.parameters()))
    for g1, g2, p in zip(
            autograd_res, bpexts_res,
            model.parameters()):
        report_nonclose_values(g1, g2)
        assert torch.allclose(
            g1, g2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_sgs(problem, device):
    problem.to(device)
    autograd_res = problem.sgs_autograd()
    bpexts_res = problem.sgs_bpexts()

    model = problem.model

    check_sizes(autograd_res, bpexts_res, list(model.parameters()))

    for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
        report_nonclose_values(g1, g2)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_ggn(problem, device):
    problem.to(device)
    model = problem.model

    autograd_res = problem.diag_ggn_autograd()
    bpexts_res = problem.diag_ggn_bpexts()

    check_sizes(autograd_res, bpexts_res, list(model.parameters()))

    for ggn1, ggn2, p in zip(autograd_res, bpexts_res,
                             model.parameters()):
        report_nonclose_values(ggn1, ggn2)
        assert torch.allclose(
            ggn1, ggn2, atol=atol, rtol=rtol)

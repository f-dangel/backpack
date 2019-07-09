import torch
import numpy as np
import pytest
from .test_problems_convolutions import TEST_PROBLEMS as CONV_TEST_PROBLEMS
from .test_problems_linear import TEST_PROBLEMS as LIN_TEST_PROBLEMS
from .test_problems_activations import TEST_PROBLEMS as ACT_TEST_PROBLEMS
from .test_problems_pooling import TEST_PROBLEMS as POOL_TEST_PROBLEMS
from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl

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

    close = np.isclose(x_numpy, y_numpy, atol=atol, rtol=rtol)
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


def check_values(list1, list2):
    for i, (g1, g2) in enumerate(zip(list1, list2)):
        print(i)
        print(g1.size())
        report_nonclose_values(g1, g2)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)


###
# Tests
###


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients(problem, device):
    problem.to(device)
    autograd_res = AutogradImpl(problem).batch_gradients()
    bpexts_res = BpextImpl(problem).batch_gradients()

    check_sizes(autograd_res, bpexts_res)
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients_sum_to_grad(problem, device):
    problem.to(device)
    autograd_res = AutogradImpl(problem).gradient()
    bpexts_batch_res = BpextImpl(problem).batch_gradients()
    bpexts_res = list([g.sum(0) for g in bpexts_batch_res])

    check_sizes(autograd_res, bpexts_res, list(problem.model.parameters()))
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_sgs(problem, device):
    problem.to(device)
    autograd_res = AutogradImpl(problem).sgs()
    bpexts_res = BpextImpl(problem).sgs()

    check_sizes(autograd_res, bpexts_res, list(problem.model.parameters()))
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_ggn(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).diag_ggn()
    bpexts_res = BpextImpl(problem).diag_ggn()

    check_sizes(autograd_res, bpexts_res, list(problem.model.parameters()))
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_l2(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).batch_l2()
    bpexts_res = BpextImpl(problem).batch_l2()

    check_sizes(autograd_res, bpexts_res)
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_variance(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).variance()
    bpexts_res = BpextImpl(problem).variance()

    check_sizes(autograd_res, bpexts_res)
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_h(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).diag_h()
    bpexts_res = BpextImpl(problem).diag_h()

    check_sizes(autograd_res, bpexts_res)
    check_values(autograd_res, bpexts_res)


@pytest.mark.parametrize(
    "problem,device", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_hmp(problem, device):
    problem.to(device)

    # create test matrices
    NUM_COLS = 10
    matrices = [
        torch.randn(p.numel(), NUM_COLS, device=device)
        for p in problem.model.parameters()
    ]

    # perform hmps on same random matrices
    autograd_res = AutogradImpl(problem).hmp(matrices)
    bpexts_res = BpextImpl(problem).hmp(matrices)

    check_sizes(autograd_res, bpexts_res)
    check_values(autograd_res, bpexts_res)

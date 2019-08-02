""" TODO: Implement all features for BN, then add to automated tests. """
import pytest

import torch

from .automated_test import atol, check_sizes, check_values, rtol
from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl
from .test_problems_bn import TEST_PROBLEMS as BN_TEST_PROBLEMS

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
    **BN_TEST_PROBLEMS,
}

ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in TEST_PROBLEMS.items():
        ALL_CONFIGURATIONS.append(tuple([prob, dev]))
        CONFIGURATION_IDS.append(probname + "-" + dev_name)


###
# Tests
###
@pytest.mark.parametrize("problem,device",
                         ALL_CONFIGURATIONS,
                         ids=CONFIGURATION_IDS)
def test_batch_gradients_sum_to_grad(problem, device):
    problem.to(device)
    backpack_batch_res = BpextImpl(problem).batch_gradients()
    backpack_res = list([g.sum(0) for g in backpack_batch_res])
    autograd_res = AutogradImpl(problem).gradient()

    check_sizes(autograd_res, backpack_res, list(problem.model.parameters()))
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem,device",
                         ALL_CONFIGURATIONS,
                         ids=CONFIGURATION_IDS)
def test_ggn_mp(problem, device):
    problem.to(device)

    NUM_COLS = 10
    matrices = [
        torch.randn(p.numel(), NUM_COLS, device=device)
        for p in problem.model.parameters()
    ]

    autograd_res = AutogradImpl(problem).ggn_mp(matrices)
    backpack_res = BpextImpl(problem).ggn_mp(matrices)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem,device",
                         ALL_CONFIGURATIONS,
                         ids=CONFIGURATION_IDS)
def test_ggn_vp(problem, device):
    problem.to(device)

    vecs = [
        torch.randn(p.numel(), device=device)
        for p in problem.model.parameters()
    ]

    backpack_res = BpextImpl(problem).ggn_vp(vecs)
    autograd_res = AutogradImpl(problem).ggn_vp(vecs)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)

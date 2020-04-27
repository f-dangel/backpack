import pytest
import torch

from .automated_test import check_sizes, check_values
from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl
from .test_problems_kfacs import TEST_PROBLEMS as BATCH1_PROBLEMS

if torch.cuda.is_available():
    DEVICES = {
        "cpu": "cpu",
        "gpu": "cuda:0",
    }
else:
    DEVICES = {
        "cpu": "cpu",
    }

BATCH1_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in BATCH1_PROBLEMS.items():
        BATCH1_CONFIGURATIONS.append((prob, dev))
        CONFIGURATION_IDS.append(probname + "-" + dev_name)


###
# Tests
###
@pytest.mark.parametrize("problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_kfra_should_equal_ggn(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).kfra_blocks()
    autograd_res = AutogradImpl(problem).ggn_blocks()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_kflr_should_equal_ggn(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).kflr_blocks()
    autograd_res = AutogradImpl(problem).ggn_blocks()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_hbp_ggn_mode_should_equal_ggn(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).hbp_single_sample_ggn_blocks()
    autograd_res = AutogradImpl(problem).ggn_blocks()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_hbp_h_mode_should_equal_h(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).hbp_single_sample_h_blocks()
    autograd_res = AutogradImpl(problem).h_blocks()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.montecarlo
@pytest.mark.parametrize("problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_kfac_should_approx_ggn_montecarlo(problem, device):
    problem.to(device)

    torch.manual_seed(0)
    autograd_res = AutogradImpl(problem).ggn_blocks()

    backpack_average_res = []
    for param_res in autograd_res:
        backpack_average_res.append(torch.zeros_like(param_res))

    mc_samples = 200
    for _ in range(mc_samples):
        backpack_res = BpextImpl(problem).kfac_blocks()
        for i, param_res in enumerate(backpack_res):
            backpack_average_res[i] += param_res

    for i in range(len(backpack_average_res)):
        backpack_average_res[i] /= mc_samples

    check_sizes(autograd_res, backpack_average_res)
    check_values(autograd_res, backpack_average_res, atol=1e-1, rtol=1e-1)

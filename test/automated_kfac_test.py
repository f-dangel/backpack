import torch
import pytest
from .test_problems_kfacs import TEST_PROBLEMS as BATCH1_PROBLEMS
from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl

from .automated_test import check_sizes, check_values

if torch.cuda.is_available():
    DEVICES = {
        "cpu": "cpu",
        "gpu": "cuda:0",
    }
else:
    DEVICES = {
        "cpu": "cpu",
    }

BATCH1_TEST_PROBLEMS = {
    **BATCH1_PROBLEMS,
}

BATCH1_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in BATCH1_TEST_PROBLEMS.items():
        BATCH1_CONFIGURATIONS.append(tuple([prob, dev]))
        CONFIGURATION_IDS.append(probname + "-" + dev_name)


###
# Tests
###
@pytest.mark.parametrize(
    "problem,device", BATCH1_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_kfra_should_equal_ggn(problem, device):
    problem.to(device)
    # torch.autograd.set_detect_anomaly(True)

    # backpack_res = BpextImpl(problem).kfra_blocks()
    autograd_res = AutogradImpl(problem).ggn_blocks()

    # check_sizes(autograd_res, backpack_res)
    # check_values(autograd_res, backpack_res)

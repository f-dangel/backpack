import numpy as np
import pytest
import torch

from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl
from .test_problems_activations import TEST_PROBLEMS as ACT_TEST_PROBLEMS
from .test_problems_bn import TEST_PROBLEMS as BN_TEST_PROBLEMS
from .test_problems_convolutions import TEST_PROBLEMS as CONV_TEST_PROBLEMS
from .test_problems_linear import TEST_PROBLEMS as LIN_TEST_PROBLEMS
from .test_problems_padding import TEST_PROBLEMS as PAD_TEST_PROBLEMS
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

TEST_PROBLEMS_NO_BN = {
    **CONV_TEST_PROBLEMS,
    **LIN_TEST_PROBLEMS,
    **ACT_TEST_PROBLEMS,
    **POOL_TEST_PROBLEMS,
    **PAD_TEST_PROBLEMS,
}

NO_BN_CONFIGURATIONS = []
NO_BN_CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in TEST_PROBLEMS_NO_BN.items():
        NO_BN_CONFIGURATIONS.append((prob, dev))
        NO_BN_CONFIGURATION_IDS.append(probname + "-" + dev_name)

TEST_PROBLEMS_BN = {
    **BN_TEST_PROBLEMS,
}
BN_CONFIGURATIONS = []
BN_CONFIGURATION_IDS = []
for dev_name, dev in DEVICES.items():
    for probname, prob in TEST_PROBLEMS_BN.items():
        BN_CONFIGURATIONS.append((prob, dev))
        BN_CONFIGURATION_IDS.append(probname + "-" + dev_name)

ALL_CONFIGURATIONS = BN_CONFIGURATIONS + NO_BN_CONFIGURATIONS
ALL_CONFIGURATION_IDS = BN_CONFIGURATION_IDS + NO_BN_CONFIGURATION_IDS

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
        print("{} versus {}. Ratio of {}".format(x, y, y / x))


def check_sizes_and_values(*plists, atol=atol, rtol=rtol):
    check_sizes(*plists)
    list1, list2 = plists
    check_values(list1, list2, atol=atol, rtol=rtol)


def check_sizes(*plists):
    for i in range(len(plists) - 1):
        assert len(plists[i]) == len(plists[i + 1])

    for params in zip(*plists):
        for i in range(len(params) - 1):
            assert params[i].size() == params[i + 1].size()


def check_values(list1, list2, atol=atol, rtol=rtol):
    for i, (g1, g2) in enumerate(zip(list1, list2)):
        print(i)
        print(g1.size())
        report_nonclose_values(g1, g2)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)


###
# Tests
###


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_batch_gradients(problem, device):
    problem.to(device)
    backpack_res = BpextImpl(problem).batch_gradients()
    autograd_res = AutogradImpl(problem).batch_gradients()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device",
    ALL_CONFIGURATIONS,
    ids=ALL_CONFIGURATION_IDS,
)
def test_batch_gradients_sum_to_grad(problem, device):
    problem.to(device)
    backpack_batch_res = BpextImpl(problem).batch_gradients()
    backpack_res = [g.sum(0) for g in backpack_batch_res]
    autograd_res = AutogradImpl(problem).gradient()

    check_sizes(autograd_res, backpack_res, list(problem.model.parameters()))
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_sgs(problem, device):
    problem.to(device)
    autograd_res = AutogradImpl(problem).sgs()
    backpack_res = BpextImpl(problem).sgs()

    check_sizes(autograd_res, backpack_res, list(problem.model.parameters()))
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_diag_ggn(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).diag_ggn()
    autograd_res = AutogradImpl(problem).diag_ggn()

    check_sizes(autograd_res, backpack_res, list(problem.model.parameters()))
    check_values(autograd_res, backpack_res)


@pytest.mark.montecarlo
@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_diag_ggn_mc_approx_ggn_montecarlo(problem, device):
    problem.to(device)

    torch.manual_seed(0)
    bp_diagggn = BpextImpl(problem).diag_ggn()

    bp_diagggn_mc_avg = []
    for param_res in bp_diagggn:
        bp_diagggn_mc_avg.append(torch.zeros_like(param_res))

    mc_samples = 500
    for _ in range(mc_samples):
        bp_diagggn_mc = BpextImpl(problem).diag_ggn_mc()
        for i, param_res in enumerate(bp_diagggn_mc):
            bp_diagggn_mc_avg[i] += param_res

    for i in range(len(bp_diagggn_mc_avg)):
        bp_diagggn_mc_avg[i] /= mc_samples

    check_sizes(bp_diagggn, bp_diagggn_mc_avg)
    check_values(bp_diagggn, bp_diagggn_mc_avg, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_batch_l2(problem, device):
    problem.to(device)

    backpack_res = BpextImpl(problem).batch_l2()
    autograd_res = AutogradImpl(problem).batch_l2()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_variance(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).variance()
    backpack_res = BpextImpl(problem).variance()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_diag_h(problem, device):
    problem.to(device)

    autograd_res = AutogradImpl(problem).diag_h()
    backpack_res = BpextImpl(problem).diag_h()

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device",
    ALL_CONFIGURATIONS,
    ids=ALL_CONFIGURATION_IDS,
)
def test_hmp(problem, device):
    problem.to(device)

    NUM_COLS = 10
    matrices = [
        torch.randn(NUM_COLS, *p.shape, device=device)
        for p in problem.model.parameters()
    ]

    backpack_res = BpextImpl(problem).hmp(matrices)
    autograd_res = AutogradImpl(problem).hmp(matrices)

    check_sizes(autograd_res, backpack_res)
    atol = 5e-4
    rtol = 5e-4
    check_values(autograd_res, backpack_res, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "problem,device",
    ALL_CONFIGURATIONS,
    ids=ALL_CONFIGURATION_IDS,
)
def test_ggn_mp(problem, device):
    problem.to(device)

    NUM_COLS = 10
    matrices = [
        torch.randn(NUM_COLS, *p.shape, device=device)
        for p in problem.model.parameters()
    ]

    autograd_res = AutogradImpl(problem).ggn_mp(matrices)
    backpack_res = BpextImpl(problem).ggn_mp(matrices)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device",
    ALL_CONFIGURATIONS,
    ids=ALL_CONFIGURATION_IDS,
)
def test_hvp(problem, device):
    problem.to(device)

    vecs = [torch.randn(*p.shape, device=device) for p in problem.model.parameters()]

    backpack_res = BpextImpl(problem).hvp(vecs)
    autograd_res = AutogradImpl(problem).hvp(vecs)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device",
    ALL_CONFIGURATIONS,
    ids=ALL_CONFIGURATION_IDS,
)
def test_ggn_vp(problem, device):
    problem.to(device)

    vecs = [torch.randn(*p.shape, device=device) for p in problem.model.parameters()]

    backpack_res = BpextImpl(problem).ggn_vp(vecs)
    autograd_res = AutogradImpl(problem).ggn_vp(vecs)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_kfac_shape(problem, device):
    problem.to(device)

    backpack_res = [kfac.shape for kfac in BpextImpl(problem).kfac_blocks()]
    autograd_res = [(dim, dim) for dim in AutogradImpl(problem).parameter_numels()]

    assert backpack_res == autograd_res


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_kfra_shape(problem, device):
    problem.to(device)

    backpack_res = [kfra.shape for kfra in BpextImpl(problem).kfra_blocks()]
    autograd_res = [(dim, dim) for dim in AutogradImpl(problem).parameter_numels()]

    assert backpack_res == autograd_res


@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_kflr_shape(problem, device):
    problem.to(device)

    backpack_res = [kflr.shape for kflr in BpextImpl(problem).kflr_blocks()]
    autograd_res = [(dim, dim) for dim in AutogradImpl(problem).parameter_numels()]

    assert backpack_res == autograd_res


@pytest.mark.parametrize("modify", ["abs", "clip"])
@pytest.mark.parametrize(
    "problem,device", NO_BN_CONFIGURATIONS, ids=NO_BN_CONFIGURATION_IDS
)
def test_pchmp_shape(problem, device, modify):
    problem.to(device)

    NUM_COLS = 10
    matrices = [
        torch.randn(NUM_COLS, *p.shape, device=device)
        for p in problem.model.parameters()
    ]

    backpack_res = BpextImpl(problem).pchmp(matrices, modify)
    check_sizes(matrices, backpack_res)

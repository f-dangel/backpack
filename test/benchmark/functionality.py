import pytest
import torch

from backpack import extend

from ..implementation.implementation_autograd import AutogradImpl
from ..implementation.implementation_bpext import BpextImpl
from ..test_problem import TestProblem


def make_large_linear_classification_problem():
    Ds = [784, 512, 256, 64, 10]
    model = torch.nn.Sequential(
        extend(torch.nn.Linear(Ds[0], Ds[1])),
        extend(torch.nn.Sigmoid()),
        extend(torch.nn.Linear(Ds[1], Ds[2])),
        extend(torch.nn.Sigmoid()),
        extend(torch.nn.Linear(Ds[2], Ds[3])),
        extend(torch.nn.Sigmoid()),
        extend(torch.nn.Linear(Ds[3], Ds[-1])),
    )
    N = 128
    X = torch.randn(size=(N, Ds[0]))
    Y = torch.randint(high=Ds[-1], size=(N,))
    lossfunc = extend(torch.nn.CrossEntropyLoss())
    return TestProblem(X, Y, model, lossfunc)


def make_smallest_linear_classification_problem():
    Ds = [16, 8, 3]
    model = torch.nn.Sequential(
        extend(torch.nn.Linear(Ds[0], Ds[1])),
        extend(torch.nn.Sigmoid()),
        extend(torch.nn.Linear(Ds[1], Ds[2])),
    )
    N = 16
    X = torch.randn(size=(N, Ds[0]))
    Y = torch.randint(high=Ds[-1], size=(N,))
    lossfunc = extend(torch.nn.CrossEntropyLoss())
    return TestProblem(X, Y, model, lossfunc)


def make_small_linear_classification_problem():
    Ds = [32, 16, 4]
    model = torch.nn.Sequential(
        extend(torch.nn.Linear(Ds[0], Ds[1])),
        extend(torch.nn.Sigmoid()),
        extend(torch.nn.Linear(Ds[1], Ds[2])),
    )
    N = 32
    X = torch.randn(size=(N, Ds[0]))
    Y = torch.randint(high=Ds[-1], size=(N,))
    lossfunc = extend(torch.nn.CrossEntropyLoss())
    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "large": make_large_linear_classification_problem(),
    "smallest": make_smallest_linear_classification_problem(),
    "small": make_small_linear_classification_problem(),
}

ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for probname, prob in reversed(list(TEST_PROBLEMS.items())):
    ALL_CONFIGURATIONS.append((prob, AutogradImpl))
    CONFIGURATION_IDS.append(probname + "-autograd")
    ALL_CONFIGURATIONS.append((prob, BpextImpl))
    CONFIGURATION_IDS.append(probname + "-bpext")


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_ggn(problem, impl, tmp_path, benchmark):
    if "large_autograd" in str(tmp_path):
        pytest.skip()
    benchmark(impl(problem).diag_ggn)


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_sgs(problem, impl, benchmark):
    benchmark(impl(problem).sgs)


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients(problem, impl, benchmark):
    benchmark(impl(problem).batch_gradients)


@pytest.mark.skip()
@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_var(problem, impl, benchmark):
    raise NotImplementedError


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_h(problem, impl, tmp_path, benchmark):
    if "large_autograd" in str(tmp_path):
        pytest.skip()
    benchmark(impl(problem).diag_h)

import torch
import pytest
from bpexts.gradient import extend
from .test_problem import TestProblem
from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl


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
    Y = torch.randint(high=Ds[-1], size=(N, ))
    lossfunc = extend(torch.nn.CrossEntropyLoss())
    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "large": make_large_linear_classification_problem()
}

ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []
for probname, prob in reversed(list(TEST_PROBLEMS.items())):
    ALL_CONFIGURATIONS.append(tuple([prob, AutogradImpl]))
    CONFIGURATION_IDS.append(probname + "-autograd")
    ALL_CONFIGURATIONS.append(tuple([prob, BpextImpl]))
    CONFIGURATION_IDS.append(probname + "-bpext")


@pytest.mark.skip(reason="Diag GGN with autograd takes way too long")
@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_diag_ggn(problem, impl, benchmark):
    benchmark(impl(problem).diag_ggn)


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_sgs(problem, impl, benchmark):
    benchmark(impl(problem).sgs)


@pytest.mark.parametrize("problem,impl", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_batch_gradients(problem, impl, benchmark):
    benchmark(impl(problem).batch_gradients)

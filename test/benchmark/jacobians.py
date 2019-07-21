import torch
import pytest

from torch.nn import Linear
from torch.nn import ReLU, Sigmoid, Tanh
from torch.nn import Conv2d, MaxPool2d, AvgPool2d
from torch.nn import Dropout
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn import Sequential

from .implementation.implementation_autograd import AutogradImpl
from .implementation.implementation_bpext import BpextImpl
from .test_problem import TestProblem

from backpack.core.layers import Flatten
from backpack import extend as xtd


N, C, H, W = 100, 3, 4, 4
D = C * H * W


def X_Y(input_type, output_type):

    if input_type is "IMAGE":
        X = torch.randn(N, C, H, W)
    elif input_type is "LINEAR":
        X = torch.randn(N, D)
    else:
        raise NotImplementedError

    if output_type is "CE":
        Y = torch.randint(high=2, size=(N, ))
    else:
        raise NotImplementedError

    return (X, Y)


models = [
    Sequential(xtd(Linear(D, 2))),
    Sequential(xtd(Linear(D, 2)), xtd(ReLU())),
    Sequential(xtd(Linear(D, 2)), xtd(Sigmoid())),
    Sequential(xtd(Linear(D, 2)), xtd(Tanh())),
    Sequential(xtd(Linear(D, 2)), xtd(Dropout())),
]
img_models = [
    Sequential(xtd(Conv2d(3, 2, 2)), Flatten(), xtd(Linear(18, 2))),
    Sequential(xtd(MaxPool2d(3)), Flatten(), xtd(Linear(3, 2))),
    Sequential(xtd(AvgPool2d(3)), Flatten(), xtd(Linear(3, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(MaxPool2d(3)), Flatten(), xtd(Linear(2, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(AvgPool2d(3)), Flatten(), xtd(Linear(2, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(ReLU()), Flatten(), xtd(Linear(18, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(Sigmoid()), Flatten(), xtd(Linear(18, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(Tanh()), Flatten(), xtd(Linear(18, 2))),
    #    Sequential(xtd(Conv2d(3, 2, 2)), xtd(Dropout()), Flatten(), xtd(Linear(18, 2))),
]


def all_problems():
    problems = []
    for model in models:
        problems.append(
            TestProblem(*X_Y("LINEAR", "CE"), model, xtd(CrossEntropyLoss())))
    for model in img_models:
        problems.append(
            TestProblem(*X_Y("IMAGE", "CE"), model, xtd(CrossEntropyLoss())))
    return problems


@pytest.mark.parametrize("problem", all_problems())
def test_all_jacobian_ag(problem, benchmark):
    benchmark(AutogradImpl(problem).gradient)


@pytest.mark.parametrize("problem", all_problems())
def test_all_jacobian_bp(problem, benchmark):
    benchmark(BpextImpl(problem).diag_ggn)

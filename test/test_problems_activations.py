import torch
from backpack import extend
from backpack.core.layers import Flatten
from .test_problem import TestProblem
from .test_problems_linear import (LINEARS, TEST_SETTINGS, linearlayer,
                                   summationLinearLayer)

ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'Sigmoid': torch.nn.Sigmoid,
    'Tanh': torch.nn.Tanh
}


def activation_layer(activation_cls):
    return extend(activation_cls())


input_size = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])
X = torch.randn(size=input_size)


def make_regression_problem(act_cls, lin_cls):
    model = torch.nn.Sequential(
        linearlayer(lin_cls, TEST_SETTINGS), activation_layer(act_cls),
        summationLinearLayer(lin_cls, TEST_SETTINGS))

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(act_cls, lin_cls):
    model = torch.nn.Sequential(
        linearlayer(lin_cls, TEST_SETTINGS), activation_layer(act_cls),
        Flatten())

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for act_name, act in ACTIVATIONS.items():
    for lin_name, lin in LINEARS.items():
        TEST_PROBLEMS["{}{}-regression".format(
            lin_name, act_name)] = make_regression_problem(act, lin)
        TEST_PROBLEMS["{}{}-classification".format(
            lin_name, act_name)] = make_classification_problem(act, lin)

import torch
from backpack import extend
from backpack.core.layers import Flatten
from .test_problem import TestProblem
from .test_problems_linear import (LINEARS, TEST_SETTINGS, linearlayer,
                                   linearlayer2, summationLinearLayer, get_X)

ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'Sigmoid': torch.nn.Sigmoid,
    'Tanh': torch.nn.Tanh
}


def activation_layer(activation_cls):
    return extend(activation_cls())


def make_regression_problem(act_cls, lin_cls, settings):
    model = torch.nn.Sequential(
        linearlayer(lin_cls, settings), activation_layer(act_cls),
        summationLinearLayer(lin_cls, settings))

    X = get_X(settings)
    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(act_cls, lin_cls, settings):
    model = torch.nn.Sequential(
        linearlayer(lin_cls, settings), activation_layer(act_cls), Flatten())

    X = get_X(settings)
    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem(act_cls, lin_cls, settings):
    model = torch.nn.Sequential(
        linearlayer(lin_cls, settings), activation_layer(act_cls),
        linearlayer2(lin_cls, settings), Flatten())

    X = get_X(settings)
    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for act_name, act in ACTIVATIONS.items():
    for lin_name, lin in LINEARS.items():
        TEST_PROBLEMS["{}{}-regression".format(
            lin_name, act_name)] = make_regression_problem(
                act, lin, TEST_SETTINGS)
        TEST_PROBLEMS["{}{}-classification".format(
            lin_name, act_name)] = make_classification_problem(
                act, lin, TEST_SETTINGS)
        TEST_PROBLEMS["{}{}-2layer-classification".format(
            lin_name, act_name)] = make_2layer_classification_problem(
                act, lin, TEST_SETTINGS)

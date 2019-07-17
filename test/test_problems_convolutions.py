import numpy as np
import torch
from backpack.core.layers import Flatten, Conv2dConcat
from backpack import extend
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 3,
    "rtol": 1e-5,
    "atol": 5e-4
}

CONVS = {
    'Conv2d': torch.nn.Conv2d,
    'Conv2dConcat': Conv2dConcat,
}


def convlayer(conv_cls):
    return extend(
        conv_cls(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            padding=TEST_SETTINGS["padding"],
            bias=TEST_SETTINGS["bias"]))


def convlayer2(conv_cls):
    return extend(
        conv_cls(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            padding=TEST_SETTINGS["padding"],
            bias=TEST_SETTINGS["bias"]))


input_size = (TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def convearlayer():
    return extend(
        torch.nn.Linear(
            in_features=np.prod([
                f - TEST_SETTINGS["padding"][0]
                for f in TEST_SETTINGS["in_features"]
            ]) * TEST_SETTINGS["out_channels"],
            out_features=1))


def make_regression_problem(conv_cls):
    model = torch.nn.Sequential(convlayer(conv_cls), Flatten(), convearlayer())

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(conv_cls):
    model = torch.nn.Sequential(convlayer(conv_cls), Flatten())

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem(conv_cls):
    model = torch.nn.Sequential(
        convlayer(conv_cls), convlayer2(conv_cls), Flatten())

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for conv_name, conv_cls in CONVS.items():
    TEST_PROBLEMS["{}-regression".format(conv_name)] = make_regression_problem(
        conv_cls)
    TEST_PROBLEMS["{}-classification".format(
        conv_name)] = make_classification_problem(conv_cls)
    TEST_PROBLEMS["{}-2layer-classification".format(
        conv_name)] = make_2layer_classification_problem(conv_cls)

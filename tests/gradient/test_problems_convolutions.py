import numpy as np
import torch
from bpexts.utils import set_seeds, Flatten
from bpexts.gradient import extend
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 5e-4
}


set_seeds(0)


def convlayer():
    return extend(torch.nn.Conv2d(
        in_channels=TEST_SETTINGS["in_features"][0],
        out_channels=TEST_SETTINGS["out_channels"],
        kernel_size=TEST_SETTINGS["kernel_size"],
        padding=TEST_SETTINGS["padding"],
        bias=TEST_SETTINGS["bias"]
    ))


def convlayer2():
    return extend(torch.nn.Conv2d(
        in_channels=TEST_SETTINGS["in_features"][0],
        out_channels=TEST_SETTINGS["out_channels"],
        kernel_size=TEST_SETTINGS["kernel_size"],
        padding=TEST_SETTINGS["padding"],
        bias=TEST_SETTINGS["bias"]
    ))


input_size = (TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def linearlayer():
    return extend(torch.nn.Linear(
        in_features=np.prod(
            [f - TEST_SETTINGS["padding"][0] for f in TEST_SETTINGS["in_features"]]
        ) * TEST_SETTINGS["out_channels"],
        out_features=1
    ))


def make_regression_problem():
    model = torch.nn.Sequential(
        convlayer(),
        Flatten(),
        linearlayer()
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem():
    model = torch.nn.Sequential(
        convlayer(),
        Flatten()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem():
    model = torch.nn.Sequential(
        convlayer(),
        convlayer2(),
        Flatten()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "conv-regression": make_regression_problem(),
    "conv-classification": make_classification_problem(),
    "conv-classification-2layer": make_2layer_classification_problem(),
}

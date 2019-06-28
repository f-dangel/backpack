import numpy as np
import torch
from bpexts.utils import set_seeds, Flatten
import bpexts.gradient.config as config
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
    return config.extend(torch.nn.Conv2d(
        in_channels=TEST_SETTINGS["in_features"][0],
        out_channels=TEST_SETTINGS["out_channels"],
        kernel_size=TEST_SETTINGS["kernel_size"],
        padding=TEST_SETTINGS["padding"],
        bias=TEST_SETTINGS["bias"]
    ))


def pooling():
    return config.extend(torch.nn.MaxPool2d(3))


input_size = (TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def linearlayer():
    return config.extend(torch.nn.Linear(
        in_features=6,
        out_features=1
    ))


def make_regression_problem():
    model = torch.nn.Sequential(
        convlayer(),
        pooling(),
        Flatten(),
        linearlayer()
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = config.extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem():
    model = torch.nn.Sequential(
        convlayer(),
        pooling(),
        Flatten()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = config.extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem():
    model = torch.nn.Sequential(
        convlayer(),
        pooling(),
        convlayer(),
        pooling(),
        Flatten()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = config.extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "conv+pooling-regression": make_regression_problem(),
    "conv+pooling-classification": make_classification_problem(),
    "conv+pooling-classification-2layer": make_2layer_classification_problem(),
}

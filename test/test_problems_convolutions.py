import numpy as np
import torch

from backpack import extend

from .layers import ACTIVATIONS, CONVS
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 3,
    "rtol": 1e-5,
    "atol": 5e-4,
}


def convlayer(conv_cls, settings):
    return extend(
        conv_cls(
            in_channels=settings["in_features"][0],
            out_channels=settings["out_channels"],
            kernel_size=settings["kernel_size"],
            padding=settings["padding"],
            bias=settings["bias"],
        )
    )


def convlayer2(conv_cls, settings):
    return extend(
        conv_cls(
            in_channels=settings["in_features"][0],
            out_channels=settings["out_channels"],
            kernel_size=settings["kernel_size"],
            padding=settings["padding"],
            bias=settings["bias"],
        )
    )


input_size = (TEST_SETTINGS["batch"],) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def convearlayer(settings):
    return extend(
        torch.nn.Linear(
            in_features=np.prod(
                [f - settings["padding"][0] for f in settings["in_features"]]
            )
            * settings["out_channels"],
            out_features=1,
        )
    )


def make_regression_problem(conv_cls, act_cls):
    model = torch.nn.Sequential(
        convlayer(conv_cls, TEST_SETTINGS),
        act_cls(),
        torch.nn.Flatten(),
        convearlayer(TEST_SETTINGS),
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(conv_cls, act_cls):
    model = torch.nn.Sequential(
        convlayer(conv_cls, TEST_SETTINGS), act_cls(), torch.nn.Flatten()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem(conv_cls, act_cls):
    model = torch.nn.Sequential(
        convlayer(conv_cls, TEST_SETTINGS),
        act_cls(),
        convlayer2(conv_cls, TEST_SETTINGS),
        act_cls(),
        torch.nn.Flatten(),
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for conv_name, conv_cls in CONVS.items():
    for act_name, act_cls in ACTIVATIONS.items():
        TEST_PROBLEMS[
            "{}-{}-regression".format(conv_name, act_name)
        ] = make_regression_problem(conv_cls, act_cls)
        TEST_PROBLEMS[
            "{}-{}-classification".format(conv_name, act_name)
        ] = make_classification_problem(conv_cls, act_cls)
        TEST_PROBLEMS[
            "{}-{}-2layer-classification".format(conv_name, act_name)
        ] = make_2layer_classification_problem(conv_cls, act_cls)

import torch

from backpack import extend

from .layers import POOLINGS
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 5e-4,
    "pool_size": 2,
}


def convlayer():
    return extend(
        torch.nn.Conv2d(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            padding=TEST_SETTINGS["padding"],
            bias=TEST_SETTINGS["bias"],
        )
    )


input_size = (TEST_SETTINGS["batch"],) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def linearlayer():
    return extend(torch.nn.Linear(in_features=18, out_features=1))


def pooling(pooling_cls):
    return extend(pooling_cls(TEST_SETTINGS["pool_size"]))


def make_regression_problem(pooling_cls):
    model = torch.nn.Sequential(
        convlayer(), pooling(pooling_cls), torch.nn.Flatten(), linearlayer()
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(pooling_cls):
    model = torch.nn.Sequential(convlayer(), pooling(pooling_cls), torch.nn.Flatten())

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem(pooling_cls):
    model = torch.nn.Sequential(
        convlayer(),
        pooling(pooling_cls),
        convlayer(),
        pooling(pooling_cls),
        torch.nn.Flatten(),
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for pool_name, pool_cls in POOLINGS.items():
    TEST_PROBLEMS["conv+{}-regression".format(pool_name)] = make_regression_problem(
        pool_cls
    )
    TEST_PROBLEMS[
        "conv+{}-classification".format(pool_name)
    ] = make_classification_problem(pool_cls)
    TEST_PROBLEMS[
        "conv+{}-classification-2layer".format(pool_name)
    ] = make_2layer_classification_problem(pool_cls)

import torch
from bpexts.utils import set_seeds
import bpexts.gradient.config as config
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 6,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 5e-4
}


set_seeds(0)
convlayer = config.extend(torch.nn.Conv2d(
    in_channels=TEST_SETTINGS["in_features"][0],
    out_channels=TEST_SETTINGS["out_channels"],
    kernel_size=TEST_SETTINGS["kernel_size"],
    padding=TEST_SETTINGS["padding"],
    bias=TEST_SETTINGS["bias"]
))

input_size = (TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def make_regression_problem():
    class SumAll(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return input.sum(dim=list(range(1, len(input.shape)))).unsqueeze(1)

    model = torch.nn.Sequential(
        convlayer,
        SumAll()
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = config.extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem():
    class To2D(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return input.view(input.shape[0], -1)

    model = torch.nn.Sequential(
        convlayer,
        To2D()
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = config.extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "conv-regression": make_regression_problem(),
    "conv-classification": make_classification_problem(),
}

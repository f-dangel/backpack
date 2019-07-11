import torch
from backpack.utils.utils import Flatten
from backpack import extend
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": 7,
    "out_features": 3,
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 1e-5
}

linearlayer = extend(
    torch.nn.Linear(
        in_features=TEST_SETTINGS["in_features"],
        out_features=TEST_SETTINGS["out_features"],
        bias=TEST_SETTINGS["bias"],
    ))
linearlayer2 = extend(
    torch.nn.Linear(
        in_features=TEST_SETTINGS["out_features"],
        out_features=3,
        bias=TEST_SETTINGS["bias"],
    ))
summationLinearLayer = extend(
    torch.nn.Linear(
        in_features=TEST_SETTINGS["out_features"], out_features=1, bias=True))
input_size = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])
X = torch.randn(size=input_size)


def make_regression_problem():
    model = torch.nn.Sequential(linearlayer, summationLinearLayer)

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem():
    model = torch.nn.Sequential(linearlayer, Flatten())

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem():
    model = torch.nn.Sequential(linearlayer, linearlayer2, Flatten())

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "linear-regression": make_regression_problem(),
    "linear-classification": make_classification_problem(),
    "linear-2layer-classification": make_2layer_classification_problem(),
}

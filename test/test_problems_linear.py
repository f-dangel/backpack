import torch
from backpack.core.layers import Flatten, LinearConcat
from backpack import extend
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": 7,
    "out_features": 3,
    "out_features2": 3,
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 1e-5
}

LINEARS = {
    'Linear': torch.nn.Linear,
    'LinearConcat': LinearConcat,
}


def linearlayer(linear_cls, settings):
    return extend(
        linear_cls(
            in_features=settings["in_features"],
            out_features=settings["out_features"],
            bias=settings["bias"],
        ))


def linearlayer2(linear_cls, settings):
    return extend(
        linear_cls(
            in_features=TEST_SETTINGS["out_features"],
            out_features=TEST_SETTINGS["out_features2"],
            bias=TEST_SETTINGS["bias"],
        ))


def summationLinearLayer(linear_cls, settings):
    return extend(
        linear_cls(
            in_features=settings["out_features"], out_features=1, bias=True))


input_size = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])
X = torch.randn(size=input_size)


def make_regression_problem(linear_cls):
    model = torch.nn.Sequential(
        linearlayer(linear_cls, TEST_SETTINGS),
        summationLinearLayer(linear_cls, TEST_SETTINGS))

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem(linear_cls):
    model = torch.nn.Sequential(
        linearlayer(linear_cls, TEST_SETTINGS), Flatten())

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


def make_2layer_classification_problem(linear_cls):
    model = torch.nn.Sequential(
        linearlayer(linear_cls, TEST_SETTINGS),
        linearlayer2(linear_cls, TEST_SETTINGS), Flatten())

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for lin_name, lin_cls in LINEARS.items():
    TEST_PROBLEMS["{}-regression".format(lin_name)] = make_regression_problem(
        lin_cls)
    TEST_PROBLEMS["{}-classification".format(
        lin_name)] = make_classification_problem(lin_cls)
    TEST_PROBLEMS["{}-2layer-classification".format(
        lin_name)] = make_2layer_classification_problem(lin_cls)

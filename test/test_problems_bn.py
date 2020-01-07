from torch.nn import BatchNorm1d

from .layers import LINEARS
from .networks import single_linear_layer, two_linear_layers
from .problems import make_classification_problem, make_regression_problem

TEST_SETTINGS = {
    "in_features": 7,
    "out_features": 3,
    "out_features2": 3,
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 1e-5,
}


def bn_layer1():
    return BatchNorm1d(num_features=TEST_SETTINGS["out_features"])


def bn_layer2():
    return BatchNorm1d(num_features=TEST_SETTINGS["out_features2"])


INPUT_SHAPE = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])

TEST_PROBLEMS = {}

for lin_name, lin_cls in LINEARS.items():

    TEST_PROBLEMS["{}-bn-regression".format(lin_name)] = make_regression_problem(
        INPUT_SHAPE,
        single_linear_layer(TEST_SETTINGS, lin_cls, activation_cls=None)
        + [BatchNorm1d(TEST_SETTINGS["out_features"])],
    )

    TEST_PROBLEMS[
        "{}-bn-classification".format(lin_name)
    ] = make_classification_problem(
        INPUT_SHAPE,
        single_linear_layer(TEST_SETTINGS, lin_cls, activation_cls=None)
        + [bn_layer1()],
    )

    TEST_PROBLEMS[
        "{}-bn-2layer-classification".format(lin_name)
    ] = make_classification_problem(
        INPUT_SHAPE,
        two_linear_layers(TEST_SETTINGS, lin_cls, activation_cls=None) + [bn_layer2()],
    )

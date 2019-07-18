from .make_problems import (make_regression_problem,
                            make_classification_problem,
                            make_2layer_classification_problem)
from .layers import ACTIVATIONS, LINEARS

TEST_SETTINGS = {
    "in_features": 7,
    "out_features": 3,
    "out_features2": 3,
    "bias": True,
    "batch": 1,
    "rtol": 1e-5,
    "atol": 1e-5
}
assert TEST_SETTINGS["batch"] == 1

REGRESSION_PROBLEMS = {}
for act_name, act in ACTIVATIONS.items():
    for lin_name, lin in LINEARS.items():
        REGRESSION_PROBLEMS["{}{}-regression".format(
            lin_name, act_name)] = make_regression_problem(
                act, lin, TEST_SETTINGS)

TEST_PROBLEMS = {
    **REGRESSION_PROBLEMS,
}
for act_name, act in ACTIVATIONS.items():
    for lin_name, lin in LINEARS.items():
        TEST_PROBLEMS["{}{}-classification".format(
            lin_name, act_name)] = make_classification_problem(
                act, lin, TEST_SETTINGS)
        TEST_PROBLEMS["{}{}-2layer-classification".format(
            lin_name, act_name)] = make_2layer_classification_problem(
                act, lin, TEST_SETTINGS)

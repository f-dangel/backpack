from .layers import LINEARS
from .make_problems import (make_regression_problem,
                            make_classification_problem,
                            make_2layer_classification_problem)

TEST_SETTINGS = {
    "in_features": 7,
    "out_features": 3,
    "out_features2": 3,
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 1e-5
}

TEST_PROBLEMS = {}

for lin_name, lin_cls in LINEARS.items():
    TEST_PROBLEMS["{}-regression".format(lin_name)] = make_regression_problem(
        None, lin_cls, TEST_SETTINGS)
    TEST_PROBLEMS["{}-classification".format(
        lin_name)] = make_classification_problem(None, lin_cls, TEST_SETTINGS)
    TEST_PROBLEMS["{}-2layer-classification".format(
        lin_name)] = make_2layer_classification_problem(
            None, lin_cls, TEST_SETTINGS)

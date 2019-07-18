import torch
from backpack.core.layers import Flatten
from backpack import extend
from .test_problem import TestProblem
from .test_problems_linear import (TEST_SETTINGS as LIN_TEST_SETTINGS, LINEARS,
                                   linearlayer, linearlayer2,
                                   summationLinearLayer)

from .test_problems_activations import (
    ACTIVATIONS, activation_layer, make_regression_problem,
    make_classification_problem, make_2layer_classification_problem)

TEST_SETTINGS = {**LIN_TEST_SETTINGS}
TEST_SETTINGS["batch"] = 1

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

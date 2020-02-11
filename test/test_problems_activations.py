from .layers import ACTIVATIONS, LINEARS
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
INPUT_SHAPE = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])

TEST_PROBLEMS = {}

for act_name, act_cls in ACTIVATIONS.items():
    for lin_name, lin_cls in LINEARS.items():

        TEST_PROBLEMS[
            "{}{}-regression".format(lin_name, act_name)
        ] = make_regression_problem(
            INPUT_SHAPE,
            single_linear_layer(TEST_SETTINGS, lin_cls, activation_cls=act_cls),
        )

        TEST_PROBLEMS[
            "{}{}-classification".format(lin_name, act_name)
        ] = make_classification_problem(
            INPUT_SHAPE,
            single_linear_layer(TEST_SETTINGS, lin_cls, activation_cls=act_cls),
        )

        TEST_PROBLEMS[
            "{}{}-2layer-classification".format(lin_name, act_name)
        ] = make_classification_problem(
            INPUT_SHAPE,
            two_linear_layers(TEST_SETTINGS, lin_cls, activation_cls=act_cls),
        )

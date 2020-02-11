import torch

from backpack import extend

from .layers import PADDINGS
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 0, 1, 1),
    "bias": True,
    "batch": 5,
    "rtol": 1e-5,
    "atol": 5e-4,
}


def conv_no_padding_layer():
    return extend(
        torch.nn.Conv2d(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            bias=TEST_SETTINGS["bias"],
        )
    )


input_size = (TEST_SETTINGS["batch"],) + TEST_SETTINGS["in_features"]
X = torch.randn(size=input_size)


def padding(padding_cls):
    return extend(padding_cls(TEST_SETTINGS["padding"]))


def make_2layer_classification_problem(padding_cls):
    model = torch.nn.Sequential(
        padding(padding_cls),
        conv_no_padding_layer(),
        padding(padding_cls),
        conv_no_padding_layer(),
        torch.nn.Flatten(),
    )

    Y = torch.randint(high=X.shape[1], size=(model(X).shape[0],))

    lossfunc = extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {}
for pad_name, pad_cls in PADDINGS.items():
    TEST_PROBLEMS[
        "conv+{}-classification-2layer".format(pad_name)
    ] = make_2layer_classification_problem(pad_cls)

"""Shared test cases for BackPACK's first-order extensions.

Shared by the tests of:
- ``BatchGrad``
- ``BatchL2Grad``
- ``SumGradSquared``
- ``Variance``

Required entries:
    "module_fn" (callable): Contains a model constructed from `torch.nn` layers
    "input_fn" (callable): Used for specifying input function
    "target_fn" (callable): Fetches the groundtruth/target classes
                            of regression/classification task
    "loss_function_fn" (callable): Loss function used in the model

Optional entries:
    "device" [list(device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
    "seed" (int): seed set before initializing a case.
"""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.automated_settings import make_simple_cnn_setting
from test.utils.evaluation_mode import initialize_training_false_recursive

from torch import device, rand, randint
from torch.nn import (
    LSTM,
    RNN,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Embedding,
    Flatten,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
)
from torchvision.models import resnet18

from backpack import convert_module_to_backpack
from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

FIRSTORDER_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "input_fn": lambda: rand(3, 10),
    "module_fn": lambda: Sequential(Linear(10, 5)),
    "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": [device("cpu")],
    "seed": 0,
    "id_prefix": "example",
}
FIRSTORDER_SETTINGS.append(example)

###############################################################################
#                         test setting: Linear Layers                         #
###############################################################################

FIRSTORDER_SETTINGS += [
    # classification
    {
        "input_fn": lambda: rand(3, 10),
        "module_fn": lambda: Sequential(Linear(10, 7), Linear(7, 5)),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: rand(3, 10),
        "module_fn": lambda: Sequential(Linear(10, 7), ReLU(), Linear(7, 5)),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    # regression
    {
        "input_fn": lambda: rand(3, 10),
        "module_fn": lambda: Sequential(Linear(10, 7), Sigmoid(), Linear(7, 5)),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 5)),
    },
]

# linear with additional dimension
FIRSTORDER_SETTINGS += [
    # regression
    {
        "input_fn": lambda: rand(3, 4, 5),
        "module_fn": lambda: Sequential(Linear(5, 3), Linear(3, 2)),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4, 2)),
        "id_prefix": "one-additional",
    },
    {
        "input_fn": lambda: rand(3, 4, 2, 5),
        "module_fn": lambda: Sequential(Linear(5, 3), Sigmoid(), Linear(3, 2)),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4, 2, 2)),
        "id_prefix": "two-additional",
    },
    {
        "input_fn": lambda: rand(3, 4, 2, 3, 5),
        "module_fn": lambda: Sequential(Linear(5, 3), Linear(3, 2)),
        "loss_function_fn": lambda: MSELoss(reduction="sum"),
        "target_fn": lambda: regression_targets((3, 4, 2, 3, 2)),
        "id_prefix": "three-additional",
    },
    # classification
    {
        "input_fn": lambda: rand(3, 4, 5),
        "module_fn": lambda: Sequential(Linear(5, 3), Linear(3, 2), Flatten()),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 8),
        "id_prefix": "one-additional",
    },
    {
        "input_fn": lambda: rand(3, 4, 2, 5),
        "module_fn": lambda: Sequential(
            Linear(5, 3), Sigmoid(), Linear(3, 2), Flatten()
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 16),
        "id_prefix": "two-additional",
    },
    {
        "input_fn": lambda: rand(3, 4, 2, 3, 5),
        "module_fn": lambda: Sequential(Linear(5, 3), ReLU(), Linear(3, 2), Flatten()),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 48),
        "id_prefix": "three-additional",
    },
]

###############################################################################
#                         test setting: Convolutional Layers                  #
"""
Syntax with default parameters:
 - `torch.nn.ConvNd(in_channels, out_channels,
    kernel_size, stride=1, padding=0, dilation=1,
    groups=1, bias=True, padding_mode='zeros)`

 - `torch.nn.ConvTransposeNd(in_channels, out_channels,
    kernel_size, stride=1, padding=0, output_padding=0,
    groups=1, bias=True, dilation=1, padding_mode='zeros)`

Note: There are 5 tests added to each `torch.nn.layers`.
For `torch.nn.ConvTranspose2d` and `torch.nn.ConvTranspose3d`
only 3 tests are added because they are very memory intensive.
"""
###############################################################################

FIRSTORDER_SETTINGS += [
    # Conv1d
    make_simple_cnn_setting((3, 3, 7), Conv1d, (3, 2, 2)),
    # test dilation & stride
    make_simple_cnn_setting((3, 2, 7), Conv1d, (2, 3, 2, 2, 0, 2)),
    # test stride & padding
    make_simple_cnn_setting((3, 3, 7), Conv1d, (3, 2, 2, 2, 1)),
    # test stride & padding & dilation
    make_simple_cnn_setting((3, 3, 8), Conv1d, (3, 6, 2, 4, 2, 3)),
    # test bias
    make_simple_cnn_setting((3, 3, 7), Conv1d, (3, 2, 2, 4, 2, 1, 1, False)),
    # Conv2d
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d, (3, 2, 2)),
    make_simple_cnn_setting((3, 2, 7, 7), Conv2d, (2, 3, 2, 2, 0, 2)),
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d, (3, 2, 2, 2, 1)),
    make_simple_cnn_setting((3, 3, 8, 8), Conv2d, (3, 6, 2, 4, 2, 3)),
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d, (3, 2, 2, 4, 2, 1, 1, False)),
    # Conv3d
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d, (3, 2, 2)),
    make_simple_cnn_setting((3, 2, 3, 7, 7), Conv3d, (2, 3, 2, 2, 0, 2)),
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d, (3, 2, 2, 3, 2)),
    make_simple_cnn_setting((3, 3, 4, 8, 8), Conv3d, (3, 6, 2, 4, 2, 3)),
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d, (3, 2, 2, 4, 2, 1, 1, False)),
    # ConvTranspose1d
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d, (3, 2, 2)),
    # test dilation & stride
    make_simple_cnn_setting((3, 2, 7), ConvTranspose1d, (2, 3, 2, 2, 0, 0, 1, True, 2)),
    # test stride & padding
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d, (3, 2, 2, 2, 1)),
    # test stride & padding & dilation
    make_simple_cnn_setting((3, 3, 8), ConvTranspose1d, (3, 6, 2, 4, 2, 0, 1, True, 3)),
    # test bias
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d, (3, 2, 2, 4, 2, 0, 1, False)),
    # ConvTranspose2d
    make_simple_cnn_setting((3, 3, 7, 7), ConvTranspose2d, (3, 2, 2)),
    make_simple_cnn_setting(
        (3, 2, 9, 9), ConvTranspose2d, (2, 4, 2, 1, 0, 0, 1, True, 2)
    ),
    make_simple_cnn_setting((3, 3, 7, 7), ConvTranspose2d, (3, 2, 2, 2, 1)),
    make_simple_cnn_setting(
        (3, 3, 7, 7), ConvTranspose2d, (3, 2, 2, 4, 2, 0, 1, False)
    ),
    # ConvTranspose3d
    make_simple_cnn_setting((3, 3, 2, 7, 7), ConvTranspose3d, (3, 2, 2)),
    make_simple_cnn_setting(
        (3, 2, 3, 5, 5), ConvTranspose3d, (2, 3, 2, 2, 2, 0, 1, True, 2)
    ),
    make_simple_cnn_setting(
        (3, 3, 2, 7, 7), ConvTranspose3d, (3, 2, 2, 4, 2, 0, 1, False)
    ),
]

###############################################################################
#                         test setting: BatchNorm                             #
###############################################################################
FIRSTORDER_SETTINGS += [
    {
        "input_fn": lambda: rand(2, 3, 4),
        "module_fn": lambda: initialize_training_false_recursive(
            BatchNorm1d(num_features=3)
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((2, 4), 3),
    },
    {
        "input_fn": lambda: rand(3, 2, 4, 3),
        "module_fn": lambda: initialize_training_false_recursive(
            BatchNorm2d(num_features=2)
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3, 4, 3), 2),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            BatchNorm3d(num_features=3)
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3, 4, 1, 2), 3),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(
                BatchNorm3d(num_features=3),
                Linear(2, 3),
                BatchNorm3d(num_features=3),
                ReLU(),
                BatchNorm3d(num_features=3),
            )
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3, 4, 1, 3), 3),
    },
]
###############################################################################
#                         test setting: RNN Layers                            #
###############################################################################
FIRSTORDER_SETTINGS += [
    {
        "input_fn": lambda: rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            RNN(input_size=6, hidden_size=3, batch_first=True),
            ReduceTuple(index=0),
            Permute(0, 2, 1),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((8, 5), 3),
    },
    {
        "input_fn": lambda: rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            RNN(input_size=6, hidden_size=3, batch_first=True),
            ReduceTuple(index=0),
            Permute(0, 2, 1),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
    {
        "input_fn": lambda: rand(4, 5, 3),
        "module_fn": lambda: Sequential(
            LSTM(3, 4, batch_first=True),
            ReduceTuple(index=0),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 20),
    },
]
###############################################################################
#                         test setting: Embedding                             #
###############################################################################
FIRSTORDER_SETTINGS += [
    {
        "input_fn": lambda: randint(0, 5, (6,)),
        "module_fn": lambda: Sequential(
            Embedding(5, 3),
            Linear(3, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((6,), 4),
    },
    {
        "input_fn": lambda: randint(0, 3, (4, 2, 2)),
        "module_fn": lambda: Sequential(
            Embedding(3, 5),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((4,), 2 * 5),
    },
]

###############################################################################
#                     test setting: torchvision resnet                        #
###############################################################################
FIRSTORDER_SETTINGS += [
    {
        "input_fn": lambda: rand(2, 3, 7, 7),
        "module_fn": lambda: convert_module_to_backpack(
            resnet18(num_classes=4).eval(), True
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 4)),
        "id_prefix": "resnet18",
    },
]

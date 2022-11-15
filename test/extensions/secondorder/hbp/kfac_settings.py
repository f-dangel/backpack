"""Define test cases for KFAC."""

from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.secondorder.secondorder_settings import (
    GROUP_CONV_SETTINGS,
    LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS,
)

from torch import rand
from torch.nn import (
    BCEWithLogitsLoss,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Flatten,
    Identity,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
)

from backpack.custom_module.branching import Parallel
from backpack.custom_module.scale_module import ScaleModule

SHARED_NOT_SUPPORTED_SETTINGS = (
    GROUP_CONV_SETTINGS + LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS
)
LOCAL_NOT_SUPPORTED_SETTINGS = []

NOT_SUPPORTED_SETTINGS = SHARED_NOT_SUPPORTED_SETTINGS + LOCAL_NOT_SUPPORTED_SETTINGS

_BATCH_SIZE_1_NO_BRANCHING_SETTINGS = [
    {
        "input_fn": lambda: rand(1, 7),
        "module_fn": lambda: Sequential(Linear(7, 3), ReLU(), Linear(3, 1)),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
    },
    {
        "input_fn": lambda: rand(1, 5),
        "module_fn": lambda: Sequential(Linear(5, 4), Sigmoid(), Linear(4, 3)),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((1,), 3),
    },
    {
        "input_fn": lambda: rand(1, 6),
        "module_fn": lambda: Sequential(Linear(6, 4), ReLU(), Linear(4, 1)),
        "loss_function_fn": lambda: BCEWithLogitsLoss(reduction="mean"),
        "target_fn": lambda: rand(1, 1),
        "id_prefix": "non-binary-labels",
    },
    {
        "input_fn": lambda: rand(1, 6),
        "module_fn": lambda: Sequential(Linear(6, 4), ReLU(), Linear(4, 1)),
        "loss_function_fn": lambda: BCEWithLogitsLoss(reduction="sum"),
        "target_fn": lambda: classification_targets(size=(1, 1), num_classes=2).float(),
        "id_prefix": "binary-labels",
    },
    # convolution with single output is a linear layer (no weight sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 4),
        "module_fn": lambda: Sequential(
            Conv1d(2, 4, 4), ReLU(), Conv1d(4, 1, 1), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "conv1d-single-output",
    },
    # convolution with single output is a linear layer (no weight sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 3, 3),
        "module_fn": lambda: Sequential(
            Conv2d(2, 4, 3), ReLU(), Conv2d(4, 1, 1), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "conv2d-single-output",
    },
    # convolution with single output is a linear layer (no weight sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 3, 3, 3),
        "module_fn": lambda: Sequential(
            Conv3d(2, 3, 3), ReLU(), Conv3d(3, 1, 1), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "conv3d-single-output",
    },
    # transpose convolution with single output is a linear layer (no weight
    # sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 9),
        "module_fn": lambda: Sequential(
            ConvTranspose1d(2, 4, 3, padding=5),
            Sigmoid(),
            ConvTranspose1d(4, 1, 1),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "convtranspose1d-single-output",
    },
    # transpose convolution with single output is a linear layer (no weight
    # sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 4, 4),
        "module_fn": lambda: Sequential(
            ConvTranspose2d(2, 3, 2, padding=2),
            Sigmoid(),
            ConvTranspose2d(3, 1, 1),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "convtranspose2d-single-output",
    },
    # transpose convolution with single output is a linear layer (no weight
    # sharing across input)
    {
        "input_fn": lambda: rand(1, 2, 6, 6, 6),
        "module_fn": lambda: Sequential(
            ConvTranspose3d(2, 4, 2, padding=3),
            Sigmoid(),
            ConvTranspose3d(4, 1, 1),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "convtranspose3d-single-output",
    },
]

_BATCH_SIZE_1_BRANCHING_SETTINGS = [
    {
        "input_fn": lambda: rand(1, 10),
        "module_fn": lambda: Sequential(
            Linear(10, 5),
            ReLU(),
            # skip connection
            Parallel(
                Identity(),
                Linear(5, 5),
            ),
            # end of skip connection
            Sigmoid(),
            Linear(5, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((1,), 4),
        "id_prefix": "branching-linear",
    },
    {
        "input_fn": lambda: rand(1, 10),
        "module_fn": lambda: Sequential(
            Linear(10, 5),
            ReLU(),
            # skip connection
            Parallel(
                ScaleModule(weight=3.0),
                Linear(5, 5),
            ),
            # end of skip connection
            Sigmoid(),
            Linear(5, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((1,), 4),
        "id_prefix": "branching-scalar",
    },
]

BATCH_SIZE_1_SETTINGS = (
    _BATCH_SIZE_1_NO_BRANCHING_SETTINGS + _BATCH_SIZE_1_BRANCHING_SETTINGS
)

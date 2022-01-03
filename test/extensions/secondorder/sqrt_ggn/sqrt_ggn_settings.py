"""Contains test settings for testing SqrtGGN extension."""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch import rand, randint
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Embedding,
    Flatten,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
)

from backpack.custom_module.pad import Pad

SQRT_GGN_SETTINGS = SECONDORDER_SETTINGS

###############################################################################
#                               Embedding                                     #
###############################################################################
SQRT_GGN_SETTINGS += [
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
        "input_fn": lambda: randint(0, 3, (3, 2, 2)),
        "module_fn": lambda: Sequential(
            Embedding(3, 2),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 2 * 2),
        "seed": 1,
    },
]

###############################################################################
#                                Custom Pad mod                               #
###############################################################################
SQRT_GGN_SETTINGS += [
    {
        "input_fn": lambda: rand(5, 3, 4, 4),
        "module_fn": lambda: Sequential(
            Conv2d(in_channels=3, out_channels=2, kernel_size=2),
            Pad((1, 1, 0, 2), mode="constant", value=0.0),
            Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=2),
            Flatten(),
            Linear(8, 2),
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((5, 2)),
    },
    {
        "input_fn": lambda: rand(5, 3, 4, 4),
        "module_fn": lambda: Sequential(
            Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1),
            ReLU(),
            Pad((1, 1, 2, 0, 1, 2), mode="constant", value=1.0),
            Conv2d(in_channels=5, out_channels=2, kernel_size=3, stride=2),
            ReLU(),
            Flatten(),
            Linear(8, 3),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((5,), 3),
    },
]

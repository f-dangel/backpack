"""Contains test settings for testing SqrtGGN extension."""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS
from test.utils.evaluation_mode import initialize_training_false_recursive

from torch import rand, randint
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    CrossEntropyLoss,
    Embedding,
    Flatten,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
)

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
#                          Batchnorm evaluation mode                          #
###############################################################################

SQRT_GGN_SETTINGS += [
    {
        "input_fn": lambda: rand(2, 3, 4),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm1d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 2, 4, 3),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm2d(num_features=2), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 2 * 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm3d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 3 * 4 * 1 * 2)),
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
                Flatten(),
            )
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 4 * 1 * 3 * 3)),
    },
]

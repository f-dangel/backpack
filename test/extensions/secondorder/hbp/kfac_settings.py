"""Define test cases for KFAC."""

from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.secondorder.secondorder_settings import (
    GROUP_CONV_SETTINGS,
    LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS,
)

from torch import rand
from torch.nn import (
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

BATCH_SIZE_1_SETTINGS = [
    {
        "input_fn": lambda: rand(1, 7),
        "module_fn": lambda: Sequential(
            Linear(7, 3), ReLU(), Flatten(start_dim=1, end_dim=-1), Linear(3, 1)
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "one-additional",
    },
    {
        "input_fn": lambda: rand(3, 10),
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
        "target_fn": lambda: classification_targets((3,), 4),
        "id_prefix": "branching-linear",
    },
    {
        "input_fn": lambda: rand(3, 10),
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
        "target_fn": lambda: classification_targets((3,), 4),
        "id_prefix": "branching-scalar",
    },
]

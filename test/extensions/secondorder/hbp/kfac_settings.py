"""Define test cases for KFAC."""

from test.extensions.secondorder.secondorder_settings import (
    GROUP_CONV_SETTINGS,
    LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS,
)

SHARED_NOT_SUPPORTED_SETTINGS = (
    GROUP_CONV_SETTINGS + LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS
)
LOCAL_NOT_SUPPORTED_SETTINGS = []

NOT_SUPPORTED_SETTINGS = SHARED_NOT_SUPPORTED_SETTINGS + LOCAL_NOT_SUPPORTED_SETTINGS

# ------------------------------------------------
from test.core.derivatives.utils import regression_targets
from torch.nn import (Sequential, Linear, Flatten, MSELoss, ReLU)
from torch import rand
BATCH_SIZE_1_SETTINGS = [
    {
        "input_fn": lambda: rand(1, 7),
        "module_fn": lambda: Sequential(Linear(7, 3), ReLU(), Flatten(start_dim=1, end_dim=-1), Linear(3, 1)),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((1, 1)),
        "id_prefix": "one-additional",
    }
]



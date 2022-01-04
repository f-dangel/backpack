"""Test configurations to test diag_h.

The tests are taken from `test.extensions.secondorder.secondorder_settings`,
but additional custom tests can be defined here by appending it to the list.
"""

from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.automated_settings import make_simple_act_setting
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch import rand
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    LogSigmoid,
    MSELoss,
    ReLU,
    Sequential,
)

from backpack.custom_module.pad import Pad

DiagHESSIAN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=True))
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=False))

###############################################################################
#                                Custom Pad mod                               #
###############################################################################
LOCAL_SETTINGS += [
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

DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

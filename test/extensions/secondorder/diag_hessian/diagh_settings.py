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

from backpack.custom_module.slicing import Slicing

DiagHESSIAN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=True))
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=False))

###############################################################################
#                              Custom Slicing mod                             #
###############################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: rand(3, 3, 4, 5),
        "module_fn": lambda: Sequential(
            Conv2d(in_channels=3, out_channels=2, kernel_size=2, padding=1),
            Slicing((slice(None), 0, slice(None, None, 2), slice(0, 2))),
            ReLU(),
            Flatten(),
            Linear(6, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 4),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 5),
        "module_fn": lambda: Sequential(
            Conv2d(in_channels=3, out_channels=2, kernel_size=2, padding=1),
            Slicing((slice(None), slice(None), 2)),
            ReLU(),
            Slicing((slice(None), 1)),
            Linear(6, 4),
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4)),
    },
]

DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

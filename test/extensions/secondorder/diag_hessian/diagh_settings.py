"""Test cases for DiagHessian and BatchDiagHessian extensions.

The tests are taken from `test.extensions.secondorder.secondorder_settings`,
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.automated_settings import (
    make_simple_act_setting,
    make_simple_pooling_setting,
)
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch.nn import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    LogSigmoid,
)

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = [
    make_simple_act_setting(LogSigmoid, bias=True),
    make_simple_act_setting(LogSigmoid, bias=False),
]

###############################################################################
#                     test setting: Adaptive Pooling Layers                   #
###############################################################################
LOCAL_SETTINGS += [
    make_simple_pooling_setting((3, 3, 7), Conv1d, AdaptiveAvgPool1d, (2,)),
    make_simple_pooling_setting((3, 3, 11, 11), Conv2d, AdaptiveAvgPool2d, (2,)),
    make_simple_pooling_setting((3, 3, 7, 7, 7), Conv3d, AdaptiveAvgPool3d, (2,)),
]


DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

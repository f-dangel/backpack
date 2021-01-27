"""Test configurations to test diag_h

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch.nn import LogSigmoid
from test.extensions.automated_settings import (
    make_simple_act_setting,
    make_simple_pooling_setting,
)

DiagHESSIAN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=True))
LOCAL_SETTINGS.append(make_simple_act_setting(LogSigmoid, bias=False))

###############################################################################
#                         test setting: Pooling Layers                       #
###############################################################################
LOCAL_SETTINGS += [
    make_simple_pooling_setting((3, 3, 7), Conv1d, AvgPool1d, (2, 1)),
    make_simple_pooling_setting((3, 3, 7), Conv1d, AvgPool1d, (2, 1, 0, True)),
    make_simple_pooling_setting((3, 3, 7), Conv1d, AvgPool1d, (2, 1, 0, False)),
    make_simple_pooling_setting((3, 3, 11, 11), Conv2d, AvgPool2d, (2, 1)),
    make_simple_pooling_setting((3, 3, 7, 7), Conv2d, AvgPool2d, (2, 1, 0, True)),
    make_simple_pooling_setting((3, 3, 7, 7), Conv2d, AvgPool2d, (2, 1, 0, False)),
    make_simple_pooling_setting((3, 3, 7, 7, 7), Conv3d, AvgPool3d, (2, 1)),
    make_simple_pooling_setting((3, 3, 7, 7, 7), Conv3d, AvgPool3d, (2, 1, 0, True)),
    make_simple_pooling_setting((3, 3, 7, 7, 7), Conv3d, AvgPool3d, (2, 1, 0, False)),
]

DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

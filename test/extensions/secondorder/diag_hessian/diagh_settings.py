"""Test cases for DiagHessian and BatchDiagHessian extensions.

The tests are taken from `test.extensions.secondorder.secondorder_settings`,
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.automated_settings import make_simple_act_setting
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch.nn import LogSigmoid

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = [
    make_simple_act_setting(LogSigmoid, bias=True),
    make_simple_act_setting(LogSigmoid, bias=False),
]

DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

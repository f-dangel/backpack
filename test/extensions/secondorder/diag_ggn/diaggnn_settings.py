"""Test configurations to test diag_ggn

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS
from torch.nn import ELU, SELU

from test.extensions.automated_settings import make_simple_cnn_setting

DiagGGN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []

###############################################################################
#                         test setting: Activation Layers                     #
###############################################################################
activations = [ELU, SELU]

for act in activations:
    for bias in [True, False]:
        LOCAL_SETTINGS.append(make_simple_cnn_setting(act, bias=bias))

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

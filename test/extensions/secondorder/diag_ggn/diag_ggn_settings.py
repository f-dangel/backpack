"""Test configurations to test diag_ggn.

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.core.derivatives.utils import regression_targets
from test.extensions.automated_settings import make_simple_act_setting
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

import torch
from torch.nn import ELU, RNN, SELU, Flatten, Sequential

from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

SHARED_SETTINGS = SECONDORDER_SETTINGS

LOCAL_SETTINGS = [
    # RNN settings
    {
        "input_fn": lambda: torch.rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(1, 0, 2),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(1, 2, 0),
            Flatten(),
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
]

###############################################################################
#                         test setting: Activation Layers                     #
###############################################################################
activations = [ELU, SELU]

for act in activations:
    for bias in [True, False]:
        LOCAL_SETTINGS.append(make_simple_act_setting(act, bias=bias))

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

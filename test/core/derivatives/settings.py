"""Test configurations for `backpack.core.derivatives`.

Required entries:
    The tests for individual categories are
    written in respective files and imported here.
    Tests:
        Activation layers
        Convolutional layers
        Linear Layers
        Loss functions
        Pooling layers
        Padding layers
"""

from test.core.derivatives.activation_settings import ACTIVATION_SETTINGS
from test.core.derivatives.convolution_settings import CONVOLUTION_SETTINGS
from test.core.derivatives.linear_settings import LINEAR_SETTINGS
from test.core.derivatives.loss_settings import LOSS_SETTINGS
from test.core.derivatives.padding_settings import PADDING_SETTINGS
from test.core.derivatives.pooling_settings import POOLING_SETTINGS

SETTINGS = []

SETTINGS.extend(
    ACTIVATION_SETTINGS
    + CONVOLUTION_SETTINGS
    + LINEAR_SETTINGS
    + LOSS_SETTINGS
    + PADDING_SETTINGS
    + POOLING_SETTINGS
)

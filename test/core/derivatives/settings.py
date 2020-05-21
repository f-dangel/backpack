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
"""

from test.core.derivatives.activation_settings import activation_SETTINGS
from test.core.derivatives.convolution_settings import convolution_SETTINGS
from test.core.derivatives.linear_settings import linear_SETTINGS
from test.core.derivatives.loss_settings import loss_SETTINGS
from test.core.derivatives.pooling_settings import pooling_SETTINGS

SETTINGS = []

SETTINGS.extend(
    activation_SETTINGS
    + convolution_SETTINGS
    + linear_SETTINGS
    + loss_SETTINGS
    + pooling_SETTINGS
)

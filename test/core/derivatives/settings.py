"""Test cases for `backpack.core.derivatives`.

Cases are divided into the following layer categories:

- Activations
- (Transposed) convolutions
- Linear
- Losses
- Padding
- Pooling
"""

from test.core.derivatives.activation_settings import ACTIVATION_SETTINGS
from test.core.derivatives.convolution_settings import CONVOLUTION_SETTINGS
from test.core.derivatives.linear_settings import LINEAR_SETTINGS
from test.core.derivatives.loss_settings import LOSS_SETTINGS
from test.core.derivatives.padding_settings import PADDING_SETTINGS
from test.core.derivatives.pooling_adaptive_settings import POOLING_ADAPTIVE_SETTINGS
from test.core.derivatives.pooling_settings import POOLING_SETTINGS

SETTINGS = (
    ACTIVATION_SETTINGS
    + CONVOLUTION_SETTINGS
    + LINEAR_SETTINGS
    + LOSS_SETTINGS
    + PADDING_SETTINGS
    + POOLING_SETTINGS
    + POOLING_ADAPTIVE_SETTINGS
)

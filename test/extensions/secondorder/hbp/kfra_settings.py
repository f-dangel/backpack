"""Define test cases for KFRA."""

from test.extensions.secondorder.hbp.kfac_settings import (
    _BATCH_SIZE_1_NO_BRANCHING_SETTINGS,
)
from test.extensions.secondorder.secondorder_settings import (
    GROUP_CONV_SETTINGS,
    LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS,
)

SHARED_NOT_SUPPORTED_SETTINGS = (
    GROUP_CONV_SETTINGS + LINEAR_ADDITIONAL_DIMENSIONS_SETTINGS
)
LOCAL_NOT_SUPPORTED_SETTINGS = []

NOT_SUPPORTED_SETTINGS = SHARED_NOT_SUPPORTED_SETTINGS + LOCAL_NOT_SUPPORTED_SETTINGS

BATCH_SIZE_1_SETTINGS = _BATCH_SIZE_1_NO_BRANCHING_SETTINGS

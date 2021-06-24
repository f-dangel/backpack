"""Test cases for the block diagonal matrix-free curvature-matrix products.

Cases are shared with the second-order extensions in
``test.extensions.secondorder.secondorder_settings`` and additional cases
can be added here by appending to ``LOCAL_SETTINGS``.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []
CURVMATPROD_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

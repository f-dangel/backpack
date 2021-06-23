"""Test cases for BackPACK extensions for the GGN diagonal.

Includes
- ``DiagGGNExact``
- ``DiagGGNMC``
- ``BatchDiagGGNExact``
- ``BatchDiagGGNMC``

Shared settings are taken from `test.extensions.secondorder.secondorder_settings`.
Additional local cases can be defined here through ``LOCAL_SETTINGS``.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

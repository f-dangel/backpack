"""Test configurations to test diag_h

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

DiagHESSIAN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTING = []

DiagHESSIAN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTING

"""Test configurations to test batch_grad

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTNGS

DiagGGN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTNGS
LOCAL_SETTING = []

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTING

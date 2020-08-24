"""Test configurations to test variance

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

VARIANCE_SETTINGS = []

SHARED_SETTINGS = FIRSTORDER_SETTINGS
LOCAL_SETTINGS = []

VARIANCE_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS

"""Test configurations to test batch_l2_grad

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

BATCHl2GRAD_SETTINGS = []

SHARED_SETTING = FIRSTORDER_SETTINGS
LOCAL_SETTING = []

BATCHl2GRAD_SETTINGS = SHARED_SETTING + LOCAL_SETTING

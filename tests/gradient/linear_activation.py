"""Test extended backpropagation for linear layer, followed by elementwise activation."""

from torch import Tensor
from torch.nn import Sequential
from bpexts.gradient.linear import Linear as G_Linear
from bpexts.gradient.sigmoid import Sigmoid as G_Sigmoid
from bpexts.gradient.relu import ReLU as G_ReLU
import bpexts.gradient.config as config
from bpexts.utils import torch_allclose as allclose, set_seeds
from .gradient_test import set_up_gradient_tests

TEST_SETTINGS = {
    "in_features": 20,
    "out_features": 10,
    "bias": True,
    "batch": 13,
    "rtol": 1e-5,
    "atol": 1e-5
}

ACTIVATIONS = {'ReLU': G_ReLU, 'Sigmoid': G_Sigmoid}

for name, activation_cls in ACTIVATIONS.items():

    def layer_fn():
        set_seeds(0)
        layer = Sequential(
            G_Linear(
                in_features=TEST_SETTINGS["in_features"],
                out_features=TEST_SETTINGS["out_features"],
                bias=TEST_SETTINGS["bias"]), activation_cls())
        # dummy functions
        layer.clear_grad_batch = lambda: None
        layer.clear_sum_grad_squared = lambda: None
        layer.clear_diag_ggn = lambda: None
        return layer

    gradient_tests = set_up_gradient_tests(
        layer_fn,
        'Linear{}'.format(name),
        input_size=(TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"]),
        atol=TEST_SETTINGS["atol"],
        rtol=TEST_SETTINGS["rtol"])

    for name, test_cls in gradient_tests:
        exec('{} = test_cls'.format(name))
        del test_cls

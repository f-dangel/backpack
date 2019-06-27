from torch.nn import Sequential
from torch.nn import Linear, Sigmoid, ReLU
from bpexts.gradient.config import extend, extended
from .automated_tests import set_up_tests


TEST_SETTINGS = {
    "in_features": 20,
    "out_features": 10,
    "bias": True,
    "batch": 13,
    "rtol": 1e-5,
    "atol": 1e-5
}

ACTIVATIONS = {'ReLU': extended(ReLU), 'Sigmoid': extended(Sigmoid)}

for name, activation_cls in ACTIVATIONS.items():

    def layer_fn():
        layer = Sequential(
            extend(Linear(
                in_features=TEST_SETTINGS["in_features"],
                out_features=TEST_SETTINGS["out_features"],
                bias=TEST_SETTINGS["bias"])), activation_cls())
        # dummy functions
        layer.clear_grad_batch = lambda: None
        layer.clear_sum_grad_squared = lambda: None
        layer.clear_diag_ggn = lambda: None
        return layer

    gradient_tests = set_up_tests(
        layer_fn,
        'Linear{}'.format(name),
        input_size=(TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"]),
        atol=TEST_SETTINGS["atol"],
        rtol=TEST_SETTINGS["rtol"])

    for name, test_cls in gradient_tests:
        exec('{} = test_cls'.format(name))
        del test_cls

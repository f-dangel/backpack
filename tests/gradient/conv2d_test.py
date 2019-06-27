"""
Test batch gradient computation of conv2d layer.

The example is taken from
    Chellapilla: High Performance Convolutional Neural Networks
    for Document Processing (2007).
"""

from torch import (Tensor, randn)
from torch.nn import Conv2d
from random import (randint, choice)
import bpexts.gradient.config as config
from bpexts.utils import torch_allclose as allclose, set_seeds
from bpexts.gradient.extensions import Extensions as ext
from .automated_tests import set_up_tests


def ExtConv2d(*args, **kwargs):
    return config.extend(Conv2d(*args, **kwargs))


TEST_ATOL = 1e-4

###
# Problem settings
###


def make_conv_params(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, bias, kernel):
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "bias": bias,
        "kernel": kernel,
    }


def make_conv_layer(LayerClass, conv_params):
    layer = LayerClass(
        in_channels=conv_params["in_channels"],
        out_channels=conv_params["out_channels"],
        kernel_size=conv_params["kernel_size"],
        stride=conv_params["stride"],
        padding=conv_params["padding"],
        dilation=conv_params["dilation"],
        bias=conv_params["bias"])
    layer.weight.data = conv_params["kernel"]
    return layer


kernel11 = [[1, 1], [2, 2]]
kernel12 = [[1, 1], [1, 1]]
kernel13 = [[0, 1], [1, 0]]
kernel21 = [[1, 0], [0, 1]]
kernel22 = [[2, 1], [2, 1]]
kernel23 = [[1, 2], [2, 0]]
kernel = Tensor([[kernel11, kernel12, kernel13],
                 [kernel21, kernel22, kernel23]]).float()

CONV_PARAMS = make_conv_params(
    in_channels=3,
    out_channels=2,
    kernel_size=(2, 2),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    bias=False,
    kernel=kernel)

# input (1 sample)
in_feature1 = [[1, 2, 0], [1, 1, 3], [0, 2, 2]]
in_feature2 = [[0, 2, 1], [0, 3, 2], [1, 1, 0]]
in_feature3 = [[1, 2, 1], [0, 1, 3], [3, 3, 2]]
in1 = Tensor([[in_feature1, in_feature2, in_feature3]]).float()
result1 = [[14, 20], [15, 24]]
result2 = [[12, 24], [17, 26]]
out1 = Tensor([[result1, result2]]).float()

conv2d = make_conv_layer(Conv2d, CONV_PARAMS)
g_conv2d = make_conv_layer(ExtConv2d, CONV_PARAMS)

inputs = [in1]
results = [out1]


def loss_function(tensor):
    """Test loss function. Sum over squared entries."""
    return ((tensor.contiguous().view(-1))**2).sum()


def test_forward():
    """Compare forward
    Handles only single instance batch.
    """
    for input, result in zip(inputs, results):
        out_conv2d = conv2d(input)
        assert allclose(out_conv2d, result)
        out_g_conv2d = g_conv2d(input)
        assert allclose(out_g_conv2d, result)


def random_convolutions_and_inputs(
        in_channels=randint(1, 3),
        out_channels=randint(1, 3),
        kernel_size=(randint(1, 3), randint(1, 3)),
        stride=(randint(1, 3), randint(1, 3)),
        padding=(randint(0, 2), randint(0, 2)),
        dilation=(randint(1, 3), randint(1, 3)),
        bias=choice([True, False]),
        batch_size=randint(1, 3),
        in_size=(randint(8, 12), randint(8, 12))):
    """Return same torch/exts 2d conv modules and random inputs.

    Arguments can be fixed by handing them over.
    """
    kernel_shape = (out_channels, in_channels) + kernel_size
    kernel = randn(kernel_shape)
    in_shape = (batch_size, in_channels) + in_size
    input = randn(in_shape)

    conv_params = make_conv_params(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        kernel=kernel)

    conv2d = make_conv_layer(Conv2d, conv_params)
    g_conv2d = make_conv_layer(ExtConv2d, conv_params)

    if bias is True:
        bias_vals = randn(out_channels)
        conv2d.bias.data = bias_vals
        g_conv2d.bias.data = bias_vals
        assert allclose(conv2d.bias, g_conv2d.bias)
    assert allclose(conv2d.weight, g_conv2d.weight)

    return conv2d, g_conv2d, input


def compare_grads(conv2d, g_conv2d, input):
    """Feed input through nn and exts conv2d, compare bias/weight grad."""
    loss = loss_function(conv2d(input))
    loss.backward()

    loss_g = loss_function(g_conv2d(input))
    with config.bpexts(ext.BATCH_GRAD):
        loss_g.backward()

    assert allclose(g_conv2d.bias.grad, conv2d.bias.grad, atol=TEST_ATOL)
    assert allclose(g_conv2d.weight.grad, conv2d.weight.grad, atol=TEST_ATOL)
    assert allclose(
        g_conv2d.bias.grad_batch.sum(0), conv2d.bias.grad, atol=TEST_ATOL)
    assert allclose(
        g_conv2d.weight.grad_batch.sum(0), conv2d.weight.grad, atol=TEST_ATOL)


def test_random_grad(random_runs=10):
    """Compare bias gradients for a single sample."""
    for i in range(random_runs):
        conv2d, g_conv2d, input = random_convolutions_and_inputs(
            bias=True, batch_size=1)
        compare_grads(conv2d, g_conv2d, input)


def test_random_grad_batch(random_runs=10):
    """Check bias gradients for a batch."""
    for i in range(random_runs):
        conv2d, g_conv2d, input = random_convolutions_and_inputs(bias=True)
        compare_grads(conv2d, g_conv2d, input)


TEST_SETTINGS = {
    "in_features": (3, 8, 9),
    "out_channels": 10,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 13,
    "rtol": 1e-5,
    "atol": 5e-4
}


def layer_fn():
    set_seeds(0)
    return ExtConv2d(
        in_channels=TEST_SETTINGS["in_features"][0],
        out_channels=TEST_SETTINGS["out_channels"],
        kernel_size=TEST_SETTINGS["kernel_size"],
        padding=TEST_SETTINGS["padding"],
        bias=TEST_SETTINGS["bias"])


gradient_tests = set_up_tests(
    layer_fn,
    'Conv2d',
    input_size=(TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"],
    atol=TEST_SETTINGS["atol"],
    rtol=TEST_SETTINGS["rtol"])

for name, test_cls in gradient_tests:
    exec('{} = test_cls'.format(name))
    del test_cls

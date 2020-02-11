"""Test batch gradient computation of linear layer."""

from torch import Tensor, allclose
from torch.nn import Linear

import backpack.extensions as new_ext
from backpack import backpack, extend


def ExtLinear(*args, **kwargs):
    return extend(Linear(*args, **kwargs))


##
# Example problems definitions
##

# predefined weight matrix and bias
weight = Tensor([[1, 2, 3], [4, 5, 6]]).float()
bias = Tensor([7, 8]).float()
in_features, out_features = 3, 2


def make_lin_layer(LayerClass, in_features, out_features, weight, bias):
    layer = LayerClass(in_features=in_features, out_features=out_features)
    layer.weight.data = weight
    layer.bias.data = bias
    return layer


lin = make_lin_layer(Linear, in_features, out_features, weight, bias)
g_lin = make_lin_layer(ExtLinear, in_features, out_features, weight, bias)


def loss_function(tensor):
    """Test loss function. Sum over squared entries."""
    return ((tensor.view(-1)) ** 2).sum()


EXAMPLE_1 = {
    "in": Tensor([[1, 1, 1]]).float(),
    "out": Tensor([[6 + 7, 15 + 8]]).float(),
    "loss": 13 ** 2 + 23 ** 2,
    "bias_grad": Tensor([2 * 13, 2 * 23]).float(),
    "bias_grad_batch": Tensor([2 * 13, 2 * 23]).float(),
    "weight_grad": Tensor([[26, 26, 26], [46, 46, 46]]).float(),
    "weight_grad_batch": Tensor([[26, 26, 26], [46, 46, 46]]).float(),
}

EXAMPLE_2 = {
    "in": Tensor([[1, 0, 1], [0, 1, 0]]).float(),
    "out": Tensor([[4 + 7, 10 + 8], [2 + 7, 5 + 8]]).float(),
    "loss": 11 ** 2 + 18 ** 2 + 9 ** 2 + 13 ** 2,
    "bias_grad": Tensor([2 * (11 + 9), 2 * (18 + 13)]),
    "bias_grad_batch": Tensor([[2 * 11, 2 * 18], [2 * 9, 2 * 13]]).float(),
    "weight_grad": Tensor([[22, 18, 22], [36, 26, 36]]).float(),
    "weight_grad_batch": Tensor(
        [[[22, 0, 22], [36, 0, 36]], [[0, 18, 0], [0, 26, 0]]]
    ).float(),
}

EXAMPLES = [EXAMPLE_1, EXAMPLE_2]

##
# Tests
##


def test_forward():
    """Compare forward of torch.nn.Linear and backpack.
    Handles single-instance and batch mode."""
    for ex in EXAMPLES:
        input, result = ex["in"], ex["out"]

        out_lin = lin(input)
        assert allclose(out_lin, result)

        out_g_lin = g_lin(input)
        assert allclose(out_g_lin, result)


def test_losses():
    """Test output of loss function."""
    for ex in EXAMPLES:  # input, loss in zip(inputs, losses):
        loss_val = loss_function(g_lin(ex["in"]))
        assert loss_val.item() == ex["loss"]


def test_grad():
    """Test computation of bias/weight gradients."""
    for ex in EXAMPLES:
        input, b_grad, w_grad = ex["in"], ex["bias_grad"], ex["weight_grad"]

        loss = loss_function(g_lin(input))
        with backpack(new_ext.BatchGrad()):
            loss.backward()

        assert allclose(g_lin.bias.grad, b_grad)
        assert allclose(g_lin.weight.grad, w_grad)

        del g_lin.bias.grad
        del g_lin.weight.grad


def test_grad_batch():
    """Test computation of bias/weight batch gradients."""
    for ex in EXAMPLES:
        input, b_grad_batch, w_grad_batch = (
            ex["in"],
            ex["bias_grad_batch"],
            ex["weight_grad_batch"],
        )

        loss = loss_function(g_lin(input))
        with backpack(new_ext.BatchGrad()):
            loss.backward()

        assert allclose(g_lin.bias.grad_batch, b_grad_batch), "{} ≠ {}".format(
            g_lin.bias.grad_batch, b_grad_batch
        )
        assert allclose(g_lin.weight.grad_batch, w_grad_batch), "{} ≠ {}".format(
            g_lin.weight.grad_batch, w_grad_batch
        )

        del g_lin.bias.grad
        del g_lin.weight.grad

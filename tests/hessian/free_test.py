"""Check L, R, and HVP operators."""

import torch
from bpexts.utils import Flatten, set_seeds
from bpexts.hessian.exact import exact_hessian
from bpexts.hessian.free import (hessian_vector_product,
                                 transposed_jacobian_vector_product,
                                 vector_to_parameter_list)
from torch.nn import Sigmoid, Linear, ReLU, CrossEntropyLoss, Sequential

batch = 5
num_inputs = 20
num_outputs = 10
inputs = [num_inputs, 16, 13, num_outputs]


def test_data():
    """Create test input and labels."""
    x, y = torch.randn(batch, num_inputs), torch.randint(
        low=0, high=num_outputs, size=(batch, ))
    return x, y


def test_model(activation='sigmoid'):
    """Network to check"""
    activation_dict = {'sigmoid': Sigmoid, 'relu': ReLU}
    activation_cls = activation_dict[activation]
    layers = []
    for i in range(len(inputs) - 1):
        layers.append(
            Linear(
                in_features=inputs[i], out_features=inputs[i + 1], bias=True))
        if i != len(inputs) - 2:
            layers.append(activation_cls())
    return Sequential(*layers)


def test_model_piecewise_linear():
    return test_model(activation='relu')


def test_input_hessian_vector_product(num_hvps=10):
    model = test_model()
    x, y = test_data()
    x.requires_grad = True
    loss_fn = CrossEntropyLoss()
    loss = 1000 * loss_fn(model(x), y)
    # brute-force computed Hessian
    hessian = exact_hessian(loss, [x]).detach()
    for _ in range(num_hvps):
        v = torch.randn(x.size())
        hvp_free, = hessian_vector_product(loss, x, v)
        hvp_full = hessian.matmul(v.view(-1)).view_as(x)
        """
        print(hvp_full)
        print(hvp_free)
        """
        assert torch.allclose(hvp_free, hvp_full, atol=1e-7)


def test_parameter_hessian_vector_product(num_hvps=10):
    model = test_model()
    x, y = test_data()
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(model(x), y)
    # brute-force computed Hessian
    hessian = exact_hessian(loss, model.parameters()).detach()
    num_params = sum(p.numel() for p in model.parameters())
    for _ in range(num_hvps):
        v = torch.randn(num_params)
        vs = vector_to_parameter_list(v, model.parameters())
        hvp_free = hessian_vector_product(loss, list(model.parameters()), vs)
        # concatenate
        hvp_free = torch.cat([h.view(-1) for h in hvp_free])
        hvp_full = hessian.matmul(v)
        """
        print(hvp_full)
        print(hvp_free)
        for i, j in zip(hvp_free, hvp_full):
            print(i, j, torch.allclose(i, j, atol=1e-7))
            assert torch.allclose(i, j, atol=1e-7)
        """
        assert torch.allclose(hvp_free, hvp_full, atol=1e-7)

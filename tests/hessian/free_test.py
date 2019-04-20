"""Check L, R, and HVP operators."""

import torch
from bpexts.utils import Flatten, set_seeds
from bpexts.hessian.exact import exact_hessian
from bpexts.hessian.free import (hessian_vector_product,
                                 transposed_jacobian_vector_product,
                                 vector_to_parameter_list, ggn_vector_product)
from torch.nn import Sigmoid, Linear, ReLU, CrossEntropyLoss, Sequential

batch = 5
num_inputs = 20
num_outputs = 10
inputs = [num_inputs, 16, 13, num_outputs]
inputs_ggn = [num_inputs, num_outputs]


def test_data():
    """Create test input and labels."""
    x, y = torch.randn(batch, num_inputs), torch.randint(
        low=0, high=num_outputs, size=(batch, ))
    return x, y


def test_model(activation='sigmoid', inputs=inputs):
    """Fully-connected network to check."""
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
    """Piecewise-linear fully-connected model with a single layer."""
    return test_model(activation='relu', inputs=inputs_ggn)


def test_input_hessian_vector_product(num_hvps=10):
    """Check Hessian-free Hessian-vector product for the input Hessian."""
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
        assert torch.allclose(hvp_free, hvp_full, atol=1e-7)


def test_parameter_hessian_vector_product(num_hvps=10):
    """Check Hessian-free Hessian-vector product for the parameter Hessian."""
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
        assert torch.allclose(hvp_free, hvp_full, atol=1e-7)


def test_parameter_ggn_vector_product(num_hvps=10):
    """Check the Hessian-free GGN-vector product for a piecewise-linear
    single-layer network.

    Note: The full GGN is *not* equivalent to the Hessian for a piecewise-linear
    network. Only the *diagonal blocks* are the same. This is why we use a single
    layer network for this test.
    """
    model = test_model_piecewise_linear()
    x, y = test_data()
    loss_fn = CrossEntropyLoss()
    out = model(x)
    loss = loss_fn(out, y)
    # brute-force computed Hessian
    hessian = exact_hessian(loss, model.parameters()).detach()
    num_params = sum(p.numel() for p in model.parameters())
    for _ in range(num_hvps):
        v = torch.randn(num_params)
        vs = vector_to_parameter_list(v, model.parameters())
        # 1) Hessian-free
        hvp_free = hessian_vector_product(loss, list(model.parameters()), vs)
        # concatenate
        hvp_free = torch.cat([h.view(-1) for h in hvp_free])
        # 2) GGN-free
        ggn_free = ggn_vector_product(loss, out, model, vs)
        # concatenate
        ggn_free = torch.cat([g.view(-1) for g in ggn_free])
        # 3) Brute-force
        hvp_full = hessian.matmul(v)
        assert torch.allclose(hvp_free, hvp_full, atol=5e-7)
        assert torch.allclose(hvp_free, ggn_free, atol=5e-7)

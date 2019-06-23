"""Test of model architectures from Chen et al.: BDA-PCH (2018)."""

import torch
from .chen2018 import (mnist_model, hbp_mnist_model, hbp_split_mnist_model,
                       cifar10_model, hbp_cifar10_model,
                       hbp_split_cifar10_model)
from bpexts.utils import set_seeds


def test_forward_mnist_models():
    """Check same behaviour of original and HBP/split MNIST model."""
    max_blocks = 5
    input = torch.randn(2, 784)
    set_seeds(0)
    original = mnist_model()
    set_seeds(0)
    hbp = hbp_mnist_model()
    set_seeds(0)
    hbp_parallel = hbp_split_mnist_model(max_blocks)
    assert torch.allclose(original(input), hbp(input), atol=1E-5)
    assert torch.allclose(original(input), hbp_parallel(input), atol=1E-5)


def test_forward_cifar10_models():
    """Check same behaviour of original and HBP/split CIFAR-10 model."""
    max_blocks = 5
    input = torch.randn(2, 3072)
    set_seeds(0)
    original = cifar10_model()
    set_seeds(0)
    hbp = hbp_cifar10_model()
    set_seeds(0)
    hbp_parallel = hbp_split_cifar10_model(max_blocks, False, False)
    assert torch.allclose(original(input), hbp(input), atol=1E-5)
    assert torch.allclose(original(input), hbp_parallel(input), atol=1E-5)


def test_hbp_approximation_mnist_model():
    """Check correct usage of HBP approximations in MNIST model."""
    aij = [True, False]
    apj = [True, False]

    # assert correct approximations in layers
    linear_idx = [0, 2, 4, 6]
    activation_idx = [1, 3, 5]
    for i in aij:
        for p in apj:
            model = hbp_mnist_model(i, p)
            for idx in linear_idx:
                assert model[idx].uses_hbp_approximation(None, p)
            # assert correct approximations in activations
            for idx in activation_idx:
                assert model[idx].uses_hbp_approximation(i, None)


def test_hbp_approximation_split_mnist_model():
    """Check correct usage of HBP approximations in split MNIST model."""
    blocks = 10
    aij = [True, False]
    apj = [True, False]

    # assert correct approximations in layers
    linear_idx = [0, 2, 4, 6]
    activation_idx = [1, 3, 5]
    for i in aij:
        for p in apj:
            model = hbp_split_mnist_model(blocks, i, p)
            for idx in linear_idx:
                assert model[idx].uses_hbp_approximation(None, p)
            # assert correct approximations in activations
            for idx in activation_idx:
                assert model[idx].uses_hbp_approximation(i, None)


def test_hbp_approximation_cifar10_model():
    """Check correct usage of HBP approximations in CIFAR-10 model."""
    aij = [True, False]
    apj = [True, False]

    # assert correct approximations in layers
    linear_idx = [0, 2, 4, 6, 8, 10, 12, 14]
    activation_idx = [1, 3, 5, 7, 9, 11, 13]
    for i in aij:
        for p in apj:
            model = hbp_cifar10_model(i, p)
            for idx in linear_idx:
                assert model[idx].uses_hbp_approximation(None, p)
            # assert correct approximations in activations
            for idx in activation_idx:
                assert model[idx].uses_hbp_approximation(i, None)


def test_hbp_approximation_split_cifar10_model():
    """Check correct usage of HBP approximations in split CIFAR-10 model."""
    blocks = 10
    aij = [True, False]
    apj = [True, False]

    # assert correct approximations in layers
    linear_idx = [0, 2, 4, 6]
    activation_idx = [1, 3, 5]
    for i in aij:
        for p in apj:
            model = hbp_split_mnist_model(blocks, i, p)
            for idx in linear_idx:
                assert model[idx].uses_hbp_approximation(None, p)
            # assert correct approximations in activations
            for idx in activation_idx:
                assert model[idx].uses_hbp_approximation(i, None)

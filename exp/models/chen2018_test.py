"""Test of model architectures from Chen et al.: BDA-PCH (2018)."""

import torch
from bpexts.hbp.sequential import convert_torch_to_hbp
from bpexts.hbp.parallel.sequential import HBPParallelSequential
from .chen2018 import (mnist_model, cifar10_model)


def test_forward_mnist_models():
    """Check same behaviour of original and HBP/split MNIST model."""
    max_blocks = 5
    input = torch.randn(2, 784)
    original = mnist_model()
    hbp = convert_torch_to_hbp(original)
    hbp_parallel = HBPParallelSequential(max_blocks, *list(hbp.children()))
    assert torch.allclose(original(input), hbp(input), atol=1E-5)
    assert torch.allclose(original(input), hbp_parallel(input), atol=1E-5)


def test_forward_cifar10_models():
    """Check same behaviour of original and HBP/split CIFAR-10 model."""
    max_blocks = 5
    input = torch.randn(2, 3072)
    original = cifar10_model()
    hbp = convert_torch_to_hbp(original)
    hbp_parallel = HBPParallelSequential(max_blocks, *list(hbp.children()))
    assert torch.allclose(original(input), hbp(input), atol=1E-5)
    assert torch.allclose(original(input), hbp_parallel(input), atol=1E-5)

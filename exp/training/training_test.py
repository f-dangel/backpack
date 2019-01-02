"""Test training function data loaders."""


import torch
from warnings import warn
from .runner_test import training_fn_on_device


def test_is_cuda_device_cpu():
    """Test whether device is recognized as CPU."""
    training = training_fn_on_device(use_gpu=False)()
    assert not training.is_device_cuda()


def test_is_cuda_device_gpu():
    """Test whether device is recognized as CPU."""
    if torch.cuda.is_available():
        training = training_fn_on_device(use_gpu=True)()
        assert training.is_device_cuda()
    else:
        warn('Could not find CUDA device')


def test_pin_memory_in_data_loading_cpu():
    """When training on CPU, data loaders need not use pinned memory."""
    training = training_fn_on_device(use_gpu=False)()
    for loader in [training.load_test_set,
                   training.load_training_set,
                   training.load_training_loss_set]:
        assert not loader().pin_memory


def test_pin_memory_in_data_loading_gpu():
    """When training on GPU, data loaders should use pinned memory."""
    if torch.cuda.is_available():
        training = training_fn_on_device(use_gpu=True)()
        for loader in [training.load_test_set,
                       training.load_training_set,
                       training.load_training_loss_set]:
            assert loader().pin_memory
    else:
        warn('Could not find CUDA device')

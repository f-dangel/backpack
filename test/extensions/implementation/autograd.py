from test.extensions.implementation.base import ExtensionsImplementation
from warnings import warn

import torch


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def batch_grad(self):
        warn("Dummy")
        return torch.tensor([42.])

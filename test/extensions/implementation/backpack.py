from test.extensions.implementation.base import ExtensionsImplementation
from warnings import warn

import torch


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def batch_grad(self):
        warn("Dummy")
        return torch.tensor([42.])

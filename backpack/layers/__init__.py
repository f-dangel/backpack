"""
"""
from torch.nn import Module
from torch import flatten


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return flatten(input, start_dim=1, end_dim=-1)


class SkipConnection(Module):
    pass

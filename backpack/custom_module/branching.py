"""Emulating branching with modules."""

from typing import Any, OrderedDict, Tuple, Union

import torch

# for marking information backpropagated by PyTorch's autograd
from torch import Tensor
from torch.nn import Module

from backpack.custom_module.scale_module import ScaleModule

BRANCH_POINT_FIELD = "_backpack_branch_point"
MERGE_POINT_FIELD = "_backpack_merge_point"
MARKER = True


class ActiveIdentity(ScaleModule):
    """Like ``torch.nn.Identity``, but creates a new node in the computation graph."""

    def __init__(self):
        """Initialization with weight=1.0."""
        super().__init__(weight=1.0)


class Branch(torch.nn.Module):
    """Module used by BackPACK to handle branching in the computation graph.

          ↗ module1 → output1
    input → module2 → output2
          ↘ ...     → ...

    Args:
        modules (torch.nn.Module): Sequence of modules. Input will be fed
            through every of these modules.
    """

    def __init__(self, *args: Union[OrderedDict[str, Module], Module]):
        """Use interface of ``torch.nn.Sequential``. Modules are parallel sequence.

        Args:
            args: either a dictionary of Modules or a Tuple of Modules
        """
        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input: Tensor) -> Tuple[Any]:
        """Feed one input through each child module.

        Args:
            input: input tensor

        Returns:
            tuple of output tensor
        """
        return tuple(module(input) for module in self.children())


class SumModule(torch.nn.Module):
    """Module used by BackPACK to handle branch merges in the computation graph.

    module 1 ↘
    module 2 → SumModule (sum)
    ...      ↗

    """

    def forward(self, *input: Tensor) -> Tensor:
        """Sum up all inputs (a tuple of tensors).

        Args:
            input: tuple of input tensors

        Returns:
            sum of all inputs

        Raises:
            ValueError: if input is no tuple
        """
        if not isinstance(input, tuple):
            raise ValueError(f"Expecting tuple as input. Got {input.__class__}")
        return sum(input)


class Parallel(torch.nn.Module):
    """Feed the same input through a parallel sequence of modules. Sum the results.

    Used by BackPACK to emulate branched computations.

           ↗ module 1 ↘
    Branch → module 2 → SumModule (sum)
           ↘  ...     ↗

    """

    def __init__(
        self,
        *args: Union[OrderedDict[str, Module], Module],
        merge_module: Module = None,
    ):
        """Use interface of ``torch.nn.Sequential``. Modules are parallel sequence.

        Args:
            args: either dictionary of modules or tuple of modules
            merge_module: The module used for merging.
                Defaults to None, which means SumModule() is used.
        """
        super().__init__()

        self.branch = Branch(*args)
        self.merge = SumModule() if merge_module is None else merge_module

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass. Concatenation of Branch and SumModule.

        Args:
            input: module input

        Returns:
            result after results are merged again
        """
        out = self.branch(input)
        out = self.merge(*out)
        return out
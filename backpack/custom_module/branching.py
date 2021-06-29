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


def mark_branch_point(arg: Tensor) -> None:
    """Mark the input as branch entry point.

    Args:
        arg: tensor to be marked
    """
    setattr(arg, BRANCH_POINT_FIELD, MARKER)


def mark_merge_point(arg: Tensor) -> None:
    """Mark the input as merge entry point.

    Args:
        arg: tensor to be marked
    """
    setattr(arg, MERGE_POINT_FIELD, MARKER)


def is_branch_point(arg: Tensor) -> bool:
    """Return whether input is marked as branch point.

    Args:
        arg: tensor to be checked

    Returns:
        whether arg is branch point
    """
    return getattr(arg, BRANCH_POINT_FIELD, None) is MARKER


def is_merge_point(arg: Tensor) -> bool:
    """Return whether input is marked as mergepoint.

    Args:
        arg: tensor to be checked

    Returns:
        whether arg is merge point
    """
    return getattr(arg, MERGE_POINT_FIELD, None) is MARKER


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
        mark_branch_point(input)

        return tuple(module(input) for module in self.children())


class Merge(torch.nn.Module):
    """Module used by BackPACK to handle branch merges in the computation graph.

    module 1 ↘
    module 2 → Merge (sum)
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

        result = sum(input)
        mark_merge_point(result)

        return result


class Parallel(torch.nn.Module):
    """Feed the same input through a parallel sequence of modules. Sum the results.

    Used by BackPACK to emulate branched computations.

           ↗ module 1 ↘
    Branch → module 2 → Merge (sum)
           ↘  ...     ↗

    """

    def __init__(self, *args: Union[OrderedDict[str, Module], Module]):
        """Use interface of ``torch.nn.Sequential``. Modules are parallel sequence.

        Args:
            args: either dictionary of modules or tuple of modules
        """
        super().__init__()

        self.branch = Branch(*args)
        self.merge = Merge()

    def forward(self, input):
        out = self.branch(input)
        out = self.merge(*out)
        return out

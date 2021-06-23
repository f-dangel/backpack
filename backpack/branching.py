"""Emulating branching with modules."""

from collections import OrderedDict

import torch

# for marking information backpropagated by PyTorch's autograd
BRANCH_POINT_FIELD = "_backpack_branch_point"
MERGE_POINT_FIELD = "_backpack_merge_point"
MARKER = True


def mark_branch_point(arg):
    """Mark the input as branch entry point."""
    setattr(arg, BRANCH_POINT_FIELD, MARKER)


def mark_merge_point(arg):
    """Mark the input as merge entry point."""
    setattr(arg, MERGE_POINT_FIELD, MARKER)


def is_branch_point(arg):
    """Return whether input is marked as branch point."""
    return getattr(arg, BRANCH_POINT_FIELD, None) is MARKER


def is_merge_point(arg):
    """Return whether input is marked as mergepoint."""
    return getattr(arg, MERGE_POINT_FIELD, None) is MARKER


class ActiveIdentity(torch.nn.Module):
    """Like ``torch.nn.Identity``, but creates a new node in the computation graph."""

    def forward(self, input):
        return 1.0 * input


class Branch(torch.nn.Module):
    """Module used by BackPACK to handle branching in the computation graph.

          ↗ module1 → output1
    input → module2 → output2
          ↘ ...     → ...

    Args:
        modules (torch.nn.Module): Sequence of modules. Input will be fed
            through every of these modules.
    """

    def __init__(self, *args):
        """Use interface of ``torch.nn.Sequential``. Modules are parallel sequence."""
        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        """Feed one input through a set of modules."""
        mark_branch_point(input)

        return tuple(module(input) for module in self.children())


class Merge(torch.nn.Module):
    """Module used by BackPACK to handle branch merges in the computation graph.

    module 1 ↘
    module 2 → Merge (sum)
    ...      ↗

    """

    def forward(self, input):
        """Sum up all inputs (a tuple of tensors)."""
        if not isinstance(input, tuple):
            raise ValueError(f"Expecting tuple as input. Got {input.__class__}")

        result = sum(input)
        mark_merge_point(result)

        return result


class Parallel(torch.nn.Sequential):
    """Feed the same input through a parallel sequence of modules. Sum the results.

    Used by BackPACK to emulate branched computations.

           ↗ module 1 ↘
    Branch → module 2 → Merge (sum)
           ↘  ...     ↗

    """

    def __init__(self, *args):
        """Use interface of ``torch.nn.Sequential``. Modules are parallel sequence."""
        super().__init__()

        self.add_module("branch", Branch(*args))
        self.add_module("merge", Merge())

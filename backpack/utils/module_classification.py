"""Contains util function for classification of modules."""
from torch.nn import Module, Sequential
from torch.nn.modules.loss import _Loss

from backpack.custom_module.branching import Branch, Parallel
from backpack.custom_module.reduce_tuple import ReduceTuple
from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_0

if TORCH_VERSION_AT_LEAST_1_9_0:
    from torch.fx import GraphModule


def is_loss(module: Module) -> bool:
    """Return whether `module` is a `torch` loss function.

    Args:
        module: A PyTorch module.

    Returns:
        Whether `module` is a loss function.
    """
    return isinstance(module, _Loss)


def is_no_op(module: Module) -> bool:
    """Return whether the module does no operation in graph.

    Args:
        module: module

    Returns:
        whether module is no operation
    """
    no_op_modules = (Sequential, Branch, Parallel, ReduceTuple)
    if TORCH_VERSION_AT_LEAST_1_9_0:
        no_op_modules += (GraphModule,)
    return isinstance(module, no_op_modules)

"""Contains util function for classification of modules."""

from torch.fx import GraphModule
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss, Sequential
from torch.nn.modules.loss import _Loss

from backpack.custom_module.branching import Parallel, _Branch
from backpack.custom_module.reduce_tuple import ReduceTuple


def is_loss(module: Module) -> bool:
    """Return whether `module` is a `torch` loss function.

    Args:
        module: A PyTorch module.

    Returns:
        Whether `module` is a loss function.
    """
    return isinstance(module, _Loss)


def is_nll(module: Module) -> bool:
    """Return whether 'module' is an NLL loss function.

    Current NLL loss functions include MSE, CE and BCEWithLogits.

    Args:
        module: A PyTorch module.

    Returns:
        Whether 'module' is an NLL loss function
    """
    return isinstance(module, (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss))


def is_no_op(module: Module) -> bool:
    """Return whether the module does no operation in graph.

    Args:
        module: module

    Returns:
        whether module is no operation
    """
    no_op_modules = (Sequential, _Branch, Parallel, ReduceTuple, GraphModule)
    return isinstance(module, no_op_modules)

"""Contains util function for classification of modules."""
from torch.fx import GraphModule
from torch.nn import Module, Sequential
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, _Loss

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


def is_mse(module: Module) -> bool:
    """Return whether 'module' is a MSELoss function.

    Args:
        module: A PyTorch module.

    Returns:
        Whether 'module' is an MSE loss function
    """
    return isinstance(module, MSELoss)


def is_ce(module: Module) -> bool:
    """Return whether 'module' is a CrossEntropyLoss function.

    Args:
        module: A PyTorch module.

    Returns:
        Whether 'module' is a CrossEntropyloss function
    """
    return isinstance(module, CrossEntropyLoss)


def is_bce(module: Module) -> bool:
    """Return whether 'module' is a BCEWithLogitsLoss function.

    Args:
        module: A PyTorch module.

    Returns:
        Whether 'module' is a BCEWithLogits loss function
    """
    return isinstance(module, BCEWithLogitsLoss)


def is_nll(module: Module) -> bool:
    """Return whether 'module' is a NLL Loss function.

    Current NLL loss functions include MSE, BCEWithLogits,
    and CrossEntropy.

    Args:
        module: A PyTorch module.

    Returns:
        Whether 'module' is an NLL loss function
    """
    return is_bce(module) or is_ce(module) or is_mse(module)


def is_no_op(module: Module) -> bool:
    """Return whether the module does no operation in graph.

    Args:
        module: module

    Returns:
        whether module is no operation
    """
    no_op_modules = (Sequential, _Branch, Parallel, ReduceTuple, GraphModule)
    return isinstance(module, no_op_modules)

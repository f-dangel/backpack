"""Tools for initializing in evaluation mode, especially BatchNorm."""
from typing import Union

from torch import rand_like
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module


def initialize_training_false_recursive(module: Module) -> Module:
    """Initializes a module recursively in evaluation mode.

    Args:
        module: the module to initialize

    Returns:
        initialized module in evaluation mode
    """
    if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
        initialize_batch_norm_eval(module)
    else:
        for module_child in module.children():
            initialize_training_false_recursive(module_child)
    return module.train(False)


def initialize_batch_norm_eval(
    module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
) -> Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]:
    """Initializes a BatchNorm module in evaluation mode.

    Args:
        module: BatchNorm module

    Returns:
        the initialized BatchNorm module in evaluation mode
    """
    module.running_mean = rand_like(module.running_mean)
    module.running_var = rand_like(module.running_var)
    module.weight.data = rand_like(module.weight)
    module.bias.data = rand_like(module.bias)
    return module.train(False)

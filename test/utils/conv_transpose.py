"""Utility functions for testing transpose convolutions."""

from typing import List, Union

from torch import Tensor
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module


def fix_index_order_conv_transpose_weights(model: Module, kfac_mats: List[Tensor]):
    """Fix index order for expanded Kronecker proxies of transpose convolution weights.

    The Kronecker product stored in weights of transpose convolutions represents the
    Hessian approximation w.r.t. `weight.transpose(0, 1)` rather than `weight` due to
    the differing index order convention of transpose convolution. This function
    transposed the axes in the expanded Hessian approximation, such that it is w.r.t.
    `weights`. Approximations for all other parameters are left unchanged.

    Args:
        model: A (container or regular) module representing a neural network.
        kfac_mats: The expanded Kronecker approximations w.r.t. the model's parameters.

    Returns:
        Expanded Kronecker approximations w.r.t. the model's parameters, but modifies
        the matrices stemming from weights of transpose convolutions in order to
        represent the correct index order.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    conv_t_modules = _get_conv_transpose_modules(model)

    fix_idx = [_index(module.weight, params) for module in conv_t_modules]

    for idx in fix_idx:
        param = params[idx]
        C_in, C_out = param.shape[:2]
        K = param.shape[2:].numel()

        kron = kfac_mats[idx]
        kron = kron.reshape(C_out, C_in, K, C_out, C_in, K)
        kron = kron.transpose(0, 1).transpose(3, 4)
        kron = kron.reshape(param.numel(), param.numel())

        kfac_mats[idx] = kron

    return kfac_mats


def _get_conv_transpose_modules(
    model: Module,
) -> List[Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d]]:
    """Extract the transpose convolution modules into a list.

    Args:
        model: Neural network represented as a (container or regular) module.

    Returns:
        List containing the transpose convolution modules of model.
    """
    children = list(model.children())

    if len(children) > 0:
        return sum([_get_conv_transpose_modules(c) for c in children], [])
    elif isinstance(model, (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)):
        return [model]
    else:
        return []


def _index(param: Tensor, param_list: List[Tensor]) -> int:
    """Return the position of `param` in `param_list`.

    Uses the data_ptr to check equality of parameters.

    Args:
        param: The tensor whose position will be returned.
        param_list: A list containing multiple tensors.

    Returns:
        Index of `param`.
    """
    param_data_ptrs = [param.data_ptr() for param in param_list]
    return param_data_ptrs.index(param.data_ptr())

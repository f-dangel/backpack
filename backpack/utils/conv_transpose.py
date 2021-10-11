"""Utility functions for extracting transpose convolution BackPACK quantities."""

from typing import Callable, Type

import torch
from einops import rearrange
from torch import einsum
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d

from backpack.utils.conv import extract_bias_diagonal as conv_extract_bias_diagonal


def get_conv_transpose_module(N: int) -> Type[Module]:
    """Return the PyTorch module class of N-dimensional transpose convolution.

    Args:
        N: Transpose convolution dimension.

    Returns:
        Transpose convolution class.
    """
    return {
        1: ConvTranspose1d,
        2: ConvTranspose2d,
        3: ConvTranspose3d,
    }[N]


def get_conv_transpose_function(N: int) -> Callable:
    """Return the PyTorch function of N-dimensional transpose convolution.

    Args:
        N: Transpose convolution dimension.

    Returns:
        Transpose convolution function.
    """
    return {
        1: conv_transpose1d,
        2: conv_transpose2d,
        3: conv_transpose3d,
    }[N]


def get_weight_gradient_factors(input, grad_out, module):
    M, C_in = input.shape[0], input.shape[1]
    kernel_size_numel = module.weight.shape[2:].numel()

    X = unfold_by_conv_transpose(input, module).reshape(M, C_in * kernel_size_numel, -1)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")

    return X, dE_dY


def extract_weight_diagonal(module, unfolded_input, S, sum_batch=True):
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module (torch.nn.ConvTranspose1d or torch.nn.ConvTranspose2d or
            torch.nn.ConvTranspose3d ): Convolution layer for which the diagonal is
            extracted w.r.t. the weight.
        unfolded_input (torch.Tensor): Unfolded input to the transpose convolution.
        S (torch.Tensor): Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch (bool, optional): Sum out the batch dimension of the weight diagonals.
            Default value: ``True``.

    Returns:
        torch.Tensor: Per-sample weight diagonal if ``sum_batch=False`` (shape
            ``(N, module.weight.shape)`` with batch size ``N``) or summed weight
            diagonal if ``sum_batch=True`` (shape ``module.weight.shape``).
    """
    S = rearrange(S, "v n (g o) ... -> v n g o (...)", g=module.groups)
    unfolded_input = rearrange(
        unfolded_input,
        "n (g c) (k x) -> n g c k x",
        g=module.groups,
        k=module.weight.shape[2:].numel(),
    )

    JS = einsum("ngckx,vngox->vngcok", (unfolded_input, S))

    sum_dims = [0, 1] if sum_batch else [0]
    out_shape = (
        module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
    )

    weight_diagonal = JS.pow_(2).sum(sum_dims).reshape(out_shape)

    return weight_diagonal


# TODO This method applies the bias Jacobian, then squares and sums the result. Intro-
# duce base class for {Batch}DiagHessian and DiagGGN{Exact,MC} and remove this method
def extract_bias_diagonal(module, S, sum_batch=True):
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module (torch.nn.ConvTranspose1d or torch.nn.ConvTranspose2d or
            torch.nn.ConvTranspose3d ): Convolution layer for which the diagonal is
            extracted w.r.t. the bias.
        unfolded_input (torch.Tensor): Unfolded input to the transpose convolution.
        S (torch.Tensor): Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch (bool, optional): Sum out the batch dimension of the bias diagonals.
            Default value: ``True``.

    Returns:
        torch.Tensor: Per-sample bias diagonal if ``sum_batch=False`` (shape
            ``(N, module.bias.shape)`` with batch size ``N``) or summed bias
            diagonal if ``sum_batch=True`` (shape ``module.bias.shape``).
    """
    return conv_extract_bias_diagonal(module, S, sum_batch=sum_batch)


def unfold_by_conv_transpose(input, module):
    """Return the unfolded input using one-hot transpose convolution.

    Args:
        input (torch.Tensor): Input to a transpose convolution.
        module (torch.nn.ConvTranspose1d or torch.nn.ConvTranspose2d or
            torch.nn.ConvTranspose3d): Transpose convolution layer that specifies
            the hyperparameters for unfolding.

    Returns:
        torch.Tensor: Unfolded input of shape ``(N, C, K * X)`` with
            ``K = module.weight.shape[2:].numel()`` the number of kernel elements
            and ``X = module.output.shape[2:].numel()`` the number of output pixels.
    """
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = module.weight.shape[2:].numel()

    def make_weight():
        weight = torch.zeros(1, kernel_size_numel, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[0, i] = extraction.reshape(*kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        weight = weight.repeat(*repeat)
        return weight.to(module.weight.device)

    conv_dim = input.dim() - 2
    conv_transpose = get_conv_transpose_function(conv_dim)

    unfold = conv_transpose(
        input,
        make_weight().to(module.weight.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, C_in, -1)

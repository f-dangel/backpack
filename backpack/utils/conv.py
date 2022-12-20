"""Utility functions for convolution layers."""

from typing import Callable, Tuple, Type, Union
from warnings import warn

from einops import rearrange
from torch import Tensor, einsum
from torch.nn import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Module,
)
from torch.nn.functional import conv1d, conv2d, conv3d, unfold
from unfoldNd import unfoldNd


def get_conv_module(N: int) -> Type[Module]:
    """Return the PyTorch module class of N-dimensional convolution.

    Args:
        N: Convolution dimension.

    Returns:
        Convolution class.
    """
    return {
        1: Conv1d,
        2: Conv2d,
        3: Conv3d,
    }[N]


def get_conv_function(N: int) -> Callable:
    """Return the PyTorch function of N-dimensional convolution.

    Args:
        N: Convolution dimension.

    Returns:
        Convolution function.
    """
    return {
        1: conv1d,
        2: conv2d,
        3: conv3d,
    }[N]


def unfold_input(module: Union[Conv1d, Conv2d, Conv3d], input: Tensor) -> Tensor:
    """Return unfolded input to a convolution.

    Use PyTorch's ``unfold`` operation for 2d convolutions (4d input tensors),
    otherwise fall back to a custom implementation.

    Args:
        module: Convolution module whose hyperparameters are used for the unfold.
        input: Input to convolution that will be unfolded.

    Returns:
        Unfolded input.
    """
    if input.dim() == 4:
        return unfold(
            input,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
    else:
        return unfold_by_conv(input, module)


def get_weight_gradient_factors(
    input: Tensor, grad_out: Tensor, module: Union[Conv1d, Conv2d, Conv3d]
) -> Tuple[Tensor, Tensor]:
    """Return the factors for constructing the gradients w.r.t. the kernel.

    Args:
        input: Convolution layer input.
        grad_out: Gradient w.r.t. to the convolution layer output.
        module: Convolution layer.

    Returns:
        Unfolded input, output gradient with flattened spatial dimensions.
    """
    X = unfold_input(module, input)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")
    return X, dE_dY


def extract_weight_diagonal(
    module: Union[Conv1d, Conv2d, Conv3d],
    unfolded_input: Tensor,
    S: Tensor,
    sum_batch: bool = True,
) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module: Convolution layer for which the diagonal is extracted w.r.t. the weight.
        unfolded_input: Unfolded input to the convolution. Shape must follow the
            conventions of ``torch.nn.Unfold``.
        S: Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch: Sum out the batch dimension of the weight diagonals.
            Default value: ``True``.

    Returns:
        Per-sample weight diagonal if ``sum_batch=False`` (shape
        ``(N, module.weight.shape)`` with batch size ``N``) or summed weight
        diagonal if ``sum_batch=True`` (shape ``module.weight.shape``).
    """
    S = rearrange(S, "v n (g c) ... -> v n g c (...)", g=module.groups)
    unfolded_input = rearrange(unfolded_input, "n (g c) k -> n g c k", g=module.groups)

    JS = einsum("ngkl,vngml->vngmk", (unfolded_input, S))

    sum_dims = [0, 1] if sum_batch else [0]
    out_shape = (
        module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
    )

    return JS.pow_(2).sum(sum_dims).reshape(out_shape)


# TODO This method applies the bias Jacobian, then squares and sums the result. Intro-
# duce base class for {Batch}DiagHessian and DiagGGN{Exact,MC} and remove this method
def extract_bias_diagonal(
    module: Union[
        Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
    ],
    S: Tensor,
    sum_batch: bool = True,
) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the bias Jacobian.

    Args:
        module: Convolution layer for which the diagonal is extracted w.r.t. the bias.
        S: Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch: Sum out the batch dimension of the bias diagonals.
            Default value: ``True``.

    Returns:
        Per-sample bias diagonal if ``sum_batch=False`` (shape
        ``(N, module.bias.shape)`` with batch size ``N``) or summed bias
        diagonal if ``sum_batch=True`` (shape ``module.bias.shape``).
    """
    start_spatial = 3
    sum_before = list(range(start_spatial, S.dim()))
    sum_after = [0, 1] if sum_batch else [0]

    return S.sum(sum_before).pow_(2).sum(sum_after)


def unfold_by_conv(input: Tensor, module: Union[Conv1d, Conv2d, Conv3d]) -> Tensor:
    """Return the unfolded input using convolution.

    Args:
        input: Convolution layer input.
        module: Convolution layer.

    Returns:
        Unfolded input. For a 2d convolution with input of shape `[N, C_in, *, *]`
        and a kernel of shape `[_, _, K_H, K_W]`, this tensor has shape
        `[N, C_in * K_H * K_W, L]` where `L` is the output's number of patches.
    """
    return unfoldNd(
        input,
        module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )


def _grad_input_padding(
    grad_output: Tensor,
    input_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    dilation: Union[None, Tuple[int]] = None,
) -> Tuple[int, ...]:
    """Determine padding for the VJP of convolution.

    # noqa: DAR101
    # noqa: DAR201
    # noqa: DAR401

    Note:
        This function was copied from the PyTorch repository (version 1.9).
        It was removed between torch 1.12.1 and torch 1.13.
    """
    if dilation is None:
        # For backward compatibility
        warn(
            "_grad_input_padding 'dilation' argument not provided. Default of 1 is used."
        )
        dilation = [1] * len(stride)

    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError(f"input_size must have {k+2} elements (got {len(input_size)})")

    def dim_size(d):
        return (
            (grad_output.size(d + 2) - 1) * stride[d]
            - 2 * padding[d]
            + 1
            + dilation[d] * (kernel_size[d] - 1)
        )

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                f"requested an input grad size of {input_size}, but valid sizes range "
                f"from {min_sizes} to {max_sizes} (for a grad_output of "
                f"{grad_output.size()[2:]})"
            )

    return tuple(input_size[d] - min_sizes[d] for d in range(k))

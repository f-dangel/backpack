"""Utility functions for extracting transpose convolution BackPACK quantities."""

from typing import Callable, Tuple, Type, Union

from einops import rearrange
from torch import Tensor, einsum
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d
from unfoldNd import unfold_transposeNd

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


def get_weight_gradient_factors(
    input: Tensor,
    grad_out: Tensor,
    module: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
) -> Tuple[Tensor, Tensor]:
    """Return factors for computing gradients w.r.t. the kernel.

    Args:
        input: Input to the transpose convolution layer.
        grad_out: Gradient w.r.t. the transpose convolution layer's output.
        module: Transpose convolution layer.

    Returns:
        unfolded input, output gradient with flattened spatial dimensions
    """
    X = unfold_by_conv_transpose(input, module)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")

    return X, dE_dY


def extract_weight_diagonal(
    module: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
    unfolded_input: Tensor,
    S: Tensor,
    sum_batch: bool = True,
) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module: Convolution layer for which the diagonal is extracted w.r.t. the weight.
        unfolded_input: Unfolded input to the transpose convolution.
        S: Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch: Sum out the batch dimension of the weight diagonals.
            Default value: ``True``.

    Returns:
        Per-sample weight diagonal if ``sum_batch=False`` (shape
        ``(N, module.weight.shape)`` with batch size ``N``) or summed weight
        diagonal if ``sum_batch=True`` (shape ``module.weight.shape``).
    """
    S = rearrange(S, "v n (g o) ... -> v n g o (...)", g=module.groups)
    unfolded_input = rearrange(
        unfolded_input,
        "n (g c k) x -> n g c k x",
        g=module.groups,
        k=module.weight.shape[2:].numel(),
    )

    JS = einsum("ngckx,vngox->vngcok", unfolded_input, S)

    sum_dims = [0, 1] if sum_batch else [0]
    out_shape = (
        module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
    )

    return JS.pow_(2).sum(sum_dims).reshape(out_shape)


# TODO This method applies the bias Jacobian, then squares and sums the result. Intro-
# duce base class for {Batch}DiagHessian and DiagGGN{Exact,MC} and remove this method
def extract_bias_diagonal(
    module: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
    S: Tensor,
    sum_batch: bool = True,
) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

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
    return conv_extract_bias_diagonal(module, S, sum_batch=sum_batch)


def unfold_by_conv_transpose(
    input: Tensor, module: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d]
) -> Tensor:
    """Return the unfolded input using one-hot transpose convolution.

    Args:
        input: Input to a transpose convolution.
        module: Transpose convolution layer that specifies the hyperparameters for
            unfolding.

    Returns:
        Unfolded input of shape ``(N, C * K, X)`` with
        ``K = module.weight.shape[2:].numel()`` the number of kernel elements
        and ``X = module.output.shape[2:].numel()`` the number of output pixels.
    """
    return unfold_transposeNd(
        input,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        # TODO The case where output_size is specified in the forward pass of a
        # ConvTransposeNd is not handled
        output_padding=0,
        dilation=module.dilation,
    )

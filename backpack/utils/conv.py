import torch
from einops import rearrange
from torch import einsum
from torch.nn.functional import conv1d, conv2d, conv3d, unfold


def unfold_input(module, input):
    """Return unfolded input to a convolution.

    Use PyTorch's ``unfold`` operation for 2d convolutions (4d input tensors),
    otherwise fall back to a custom implementation.

    Args:
        module (torch.nn.Conv1d or torch.nn.Conv2d or torch.nn.Conv3d): Convolution
            module whose hyperparameters are used for the unfold.
        input (torch.Tensor): Input to convolution that will be unfolded.

    Returns:
        torch.Tensor: Unfolded input.
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


def get_weight_gradient_factors(input, grad_out, module, N):
    X = unfold_input(module, input)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")
    return X, dE_dY


def get_bias_gradient_factors(gradient, C_axis, N):
    if N == 1:
        bias_gradient = (einsum("ncl->nc", gradient) ** 2).sum(C_axis)
    elif N == 2:
        bias_gradient = (einsum("nchw->nc", gradient) ** 2).sum(C_axis)
    elif N == 3:
        bias_gradient = (einsum("ncdhw->nc", gradient) ** 2).sum(C_axis)
    else:
        raise ValueError("{}-dimensional Conv. is not implemented.".format(N))
    return bias_gradient


def separate_channels_and_pixels(module, tensor):
    """Reshape (V, N, C, H, W) into (V, N, C, H * W)."""
    return rearrange(tensor, "v n c ... -> v n c (...)")


def extract_weight_diagonal(module, unfolded_input, S, sum_batch=True):
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module (torch.nn.Conv1d or torch.nn.Conv2d or torch.nn.Conv3d): Convolution
            layer for which the diagonal is extracted w.r.t. the weight.
        unfolded_input (torch.Tensor): Unfolded input to the convolution. Shape must
            follow the conventions of ``torch.nn.Unfold``.
        S (torch.Tensor): Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch (bool, optional): Sum out the batch dimension of the weight diagonals.
            Default value: ``True``.

    Returns:
        torch.Tensor: Per-sample weight diagonal if ``sum_batch=False`` (shape
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

    weight_diagonal = JS.pow_(2).sum(sum_dims).reshape(out_shape)

    return weight_diagonal


# TODO This method applies the bias Jacobian, then squares and sums the result. Intro-
# duce base class for {Batch}DiagHessian and DiagGGN{Exact,MC} and remove this method
def extract_bias_diagonal(module, S, sum_batch=True):
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the bias Jacobian.

    Args:
        module (torch.nn.Conv1d or torch.nn.Conv2d or torch.nn.Conv3d): Convolution
            layer for which the diagonal is extracted w.r.t. the bias.
        S (torch.Tensor): Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        sum_batch (bool, optional): Sum out the batch dimension of the bias diagonals.
            Default value: ``True``.

    Returns:
        torch.Tensor: Per-sample bias diagonal if ``sum_batch=False`` (shape
            ``(N, module.bias.shape)`` with batch size ``N``) or summed bias
            diagonal if ``sum_batch=True`` (shape ``module.bias.shape``).
    """
    start_spatial = 3
    sum_before = list(range(start_spatial, S.dim()))
    sum_after = [0, 1] if sum_batch else [0]

    return S.sum(sum_before).pow_(2).sum(sum_after)


def unfold_by_conv(input, module):
    """Return the unfolded input using convolution"""
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = module.weight.shape[2:].numel()

    def make_weight():
        weight = torch.zeros(kernel_size_numel, 1, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[i] = extraction.reshape(1, *kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        return weight.repeat(*repeat)

    def get_conv():
        functional_for_module_cls = {
            torch.nn.Conv1d: conv1d,
            torch.nn.Conv2d: conv2d,
            torch.nn.Conv3d: conv3d,
        }
        return functional_for_module_cls[module.__class__]

    conv = get_conv()
    unfold = conv(
        input,
        make_weight().to(input.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, C_in * kernel_size_numel, -1)

import torch
from einops import rearrange
from torch import einsum
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d


def get_weight_gradient_factors(input, grad_out, module, N):
    M, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    X = unfold_by_conv_transpose(input, module).reshape(M, C_in * kernel_size_numel, -1)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")

    return X, dE_dY


def extract_weight_diagonal(module, unfolded_input, S, N, sum_batch=True):
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module (torch.nn.ConvTranspose1d or torch.nn.ConvTranspose2d or
            torch.nn.ConvTranspose3d ): Convolution layer for which the diagonal is
            extracted w.r.t. the weight.
        unfolded_input (torch.Tensor): Unfolded input to the transpose convolution.
        S (torch.Tensor): Backpropagated (symmetric factorization) of the loss Hessian.
            Has shape ``(V, *module.output.shape)``.
        N (int): Transpose convolution dimension.
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


def extract_bias_diagonal(module, sqrt, N, sum_batch=True):
    """
    `sqrt` must be the backpropagated quantity for DiagH or DiagGGN(MC)
    """
    V_axis, N_axis = 0, 1

    if N == 1:
        einsum_eq = "vncl->vnc"
    elif N == 2:
        einsum_eq = "vnchw->vnc"
    elif N == 3:
        einsum_eq = "vncdhw->vnc"
    else:
        ValueError("{}-dimensional ConvTranspose is not implemented.".format(N))
    sum_dims = [V_axis, N_axis] if sum_batch else [V_axis]
    return (einsum(einsum_eq, sqrt) ** 2).sum(sum_dims)


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
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    def make_weight():
        weight = torch.zeros(1, kernel_size_numel, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[0, i] = extraction.reshape(*kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        weight = weight.repeat(*repeat)
        return weight.to(module.weight.device)

    def get_conv_transpose():
        functional_for_module_cls = {
            torch.nn.ConvTranspose1d: conv_transpose1d,
            torch.nn.ConvTranspose2d: conv_transpose2d,
            torch.nn.ConvTranspose3d: conv_transpose3d,
        }
        return functional_for_module_cls[module.__class__]

    conv_transpose = get_conv_transpose()
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

from torch.nn import Unfold

from backpack.utils.einsum import einsum, eingroup


def unfold_func(module):
    return Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )


def get_weight_gradient_factors(input, grad_out, module):
    # shape [N, C_in * K_x * K_y, H_out * W_out]
    X = unfold_func(module)(input)
    dE_dY = eingroup("n,c,h,w->n,c,hw", grad_out)
    return X, dE_dY


def separate_channels_and_pixels(module, tensor):
    """Reshape (V, N, C, H, W) into (V, N, C, H * W)."""
    return eingroup("v,n,c,h,w->v,n,c,hw", tensor)


def extract_weight_diagonal(module, input, grad_output):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    grad_output_viewed = separate_channels_and_pixels(module, grad_output)
    AX = einsum("nkl,vnml->vnkm", (input, grad_output_viewed))
    return (AX ** 2).sum([0, 1]).transpose(0, 1)

from torch.nn import Unfold

from backpack.utils.einsum import einsum


def unfold_func(module):
    return Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )


def get_weight_gradient_factors(input, grad_out, module):
    batch = input.size(0)
    X = unfold_func(module)(input)
    dE_dY = grad_out.contiguous().view(batch, module.out_channels, -1)
    return X, dE_dY


def separate_channels_and_pixels(module, tensor, new_convention=True):
    """Reshape (batch, out_features, classes)
    into       (batch, out_channels, pixels, classes).
    """
    batch, channels, pixels, classes = (
        module.input0.size(0),
        module.out_channels,
        module.output_shape[2] * module.output_shape[3],
        -1,
    )
    if new_convention:
        return tensor.contiguous().view(classes, batch, channels, pixels)
    else:
        return tensor.contiguous().view(batch, channels, pixels, classes)


def extract_weight_diagonal(module, input, grad_output, new_convention=False):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    grad_output_viewed = separate_channels_and_pixels(
        module, grad_output, new_convention=new_convention
    )
    if new_convention:
        AX = einsum("bkl,cbml->cbkm", (input, grad_output_viewed))
        return (AX ** 2).sum([0, 1]).transpose(0, 1)
    else:
        AX = einsum("bkl,bmlc->cbkm", (input, grad_output_viewed))
        return (AX ** 2).sum([0, 1]).transpose(0, 1)

from torch.nn import Unfold
from backpack.utils.utils import einsum


def unfold_func(module):
    return Unfold(kernel_size=module.kernel_size,
                  dilation=module.dilation,
                  padding=module.padding,
                  stride=module.stride)

def get_weight_gradient_factors(input, grad_out, module):
    batch = input.size(0)
    X = unfold_func(module)(input)
    dE_dY = grad_out.contiguous().view(batch, module.out_channels, -1)
    return X, dE_dY


def separate_channels_and_pixels(module, tensor):
    """Reshape (batch, out_features, classes)
    into       (batch, out_channels, pixels, classes).
    """
    batch, channels, pixels, classes = (
        module.input0.size(0),
        module.out_channels,
        module.output_shape[2] * module.output_shape[3],
        -1,
    )
    return tensor.contiguous().view(batch, channels, pixels, classes)


def check_sizes_input_jac_t(mat, module):
    batch, out_channels, out_x, out_y = module.output_shape
    assert tuple(mat.size())[:2] == (batch, out_channels * out_x * out_y)


def check_sizes_input_jac(mat, module):
    batch, in_channels, in_x, in_y = module.input0.size()
    assert tuple(mat.size())[:2] == (batch, in_channels * in_x * in_y)


def check_sizes_output_jac_t(jtmp, module):
    if tuple(jtmp.size())[1:] != tuple(module.input0.size())[1:]:
        raise ValueError(
            "Size after conv_transpose does not match", "Got {}, and {}.",
            "Expected all dimensions to match, except for the first.".format(
                jtmp.size(), module.input0.size()))


def check_sizes_output_jac(jmp, module):
    if tuple(jmp.size())[1:] != tuple(module.output_shape)[1:]:
        raise ValueError(
            "Size after conv does not match", "Got {}, and {}.",
            "Expected all dimensions to match, except for the first.".format(
                jmp.size(), module.output_shape))

def extract_weight_diagonal(module, input, grad_output):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    grad_output_viewed = separate_channels_and_pixels(module, grad_output)
    AX = einsum('bkl,bmlc->cbkm', (input, grad_output_viewed))
    return (AX ** 2).sum([0, 1]).transpose(0, 1)

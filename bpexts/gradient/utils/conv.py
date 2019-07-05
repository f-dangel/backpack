from torch.nn import Unfold


def unfold_func(module):
    return Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride)


def get_weight_gradient_factors(input, grad_out, module):
    batch = input.size(0)
    X = unfold_func(module)(input)
    dE_dY = grad_out.view(batch, module.out_channels, -1)
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
    return tensor.view(batch, channels, pixels, classes)


def check_sizes_input(mat, module):
    batch, out_channels, out_x, out_y = module.output_shape
    assert tuple(mat.size())[:2] == (batch, out_channels * out_x * out_y)


def check_sizes_output(jmp, module):
    if tuple(jmp.size())[1:] != tuple(module.input0.size())[1:]:
        raise ValueError(
            "Size after conv_transpose does not match", "Got {}, and {}.",
            "Expected all dimensions to match, except for the first.".format(
                jmp.size(), module.input0.size()))

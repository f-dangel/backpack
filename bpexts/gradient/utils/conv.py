import torch


def unfold_func(module):
    return torch.nn.Unfold(
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

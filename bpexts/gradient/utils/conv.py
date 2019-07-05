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

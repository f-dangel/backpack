import torch


def unfold_func(module):
    return torch.nn.Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride)

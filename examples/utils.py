from torch import nn
from math import ceil

# TODO: Use the DeepOBS layers


def _determine_padding_from_tf_same(input_dimensions, kernel_dimensions,
                                    stride_dimensions):
    """Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution
     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0)
    pad_along_width = max(
        (out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'"""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride)

    return hook


class tfmaxpool2d(nn.Sequential):
    """Implements tf's padding 'same' for maxpooling"""

    def __init__(self,
                 kernel_size,
                 stride=None,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False,
                 tf_padding_type=None):

        super(tfmaxpool2d, self).__init__()

        if tf_padding_type == 'same':
            self.add_module('padding', nn.ZeroPad2d(0))
            hook = hook_factory_tf_padding_same(kernel_size, stride)
            self.padding.register_forward_pre_hook(hook)

        self.add_module(
            'maxpool',
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            ))


class tfconv2d(nn.Sequential):
    """Implements tf's padding 'same' for convolutions"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 tf_padding_type=None):

        super(tfconv2d, self).__init__()

        if tf_padding_type == 'same':
            self.add_module('padding', nn.ZeroPad2d(0))
            hook = hook_factory_tf_padding_same(kernel_size, stride)
            self.padding.register_forward_pre_hook(hook)

        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias))

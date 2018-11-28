"""Extension of torch.nn.Conv2d for computing batch gradients."""

from torch.nn import (Conv2d, Fold, Unfold)
from torch import (randn, einsum)
from numpy import prod
from ..decorator import decorate


# decorated torch.nn.Conv2d module
DecoratedConv2d = decorate(Conv2d)


class G_Conv2d(DecoratedConv2d):
    """Extended backpropagation for torch.nn.Conv2d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfold = None
        self.fold = None

    def forward(self, input):
        self.unfold = Unfold(kernel_size=self.kernel_size,
                             dilation=self.dilation,
                             padding=self.padding,
                             stride=self.stride)
        out_size = self.output_size(input.size()[1:])
        # expand patches
        im2col = self.unfold(input)
        print('im2col', im2col)
        # perform convolution by matrix multiplication
        kernel_matrix = self.weight.view(self.out_channels, -1)
        print('kernel matrix', kernel_matrix)
        # j: output image size, k: output channels
        convoluted = einsum('ki,bij->bkj', (kernel_matrix, im2col))
        print('convoluted', convoluted)
        # reshape into output image
        out_shape = (-1, self.out_channels) + out_size
        col2im = convoluted.view(out_shape)
        print('col2im', col2im)
        return col2im

    def output_size(self, input_size):
        """Compute size of the output channels from random input.

        Taken from
            https://discuss.pytorch.org/t/convolution-and-pooling-
            layers-need-a-method-to-calculate-output-size/21895/2
        """
        x = randn(input_size).unsqueeze(0)
        output = super().forward(x)
        return output.size()[2:]

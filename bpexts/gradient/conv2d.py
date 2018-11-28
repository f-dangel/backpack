"""Extension of torch.nn.Conv2d for computing batch gradients."""

from torch.nn import Conv2d
from ..decorator import decorate


# decorated torch.nn.Conv2d module
DecoratedConv2d = decorate(Conv2d)


class G_Conv2d(DecoratedConv2d):
    """Extended backpropagation for torch.nn.Conv2d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

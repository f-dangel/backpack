"""Curvature-vector products for linear layers."""

from numpy import prod
from torch import (einsum, arange, zeros, tensor)
from torch.nn import functional
from torch.nn import Conv2d
from ..hbp.module import hbp_decorate


class CVPConv2d(hbp_decorate(Conv2d)):
    """2D Convolution with recursive Hessian-vector products."""

    # override
    def hbp_hooks(self):
        """Install hook storing unfolded input."""
        self.register_exts_forward_hook(self.store_input_and_output_dimensions)

    # --- hooks ---
    @staticmethod
    def store_input_and_output_dimensions(module, input, output):
        """Save input and dimensions of the output to the layer.

        Intended use as forward hook.
        Initialize module buffer 'layer_input' and attribute
        'output_size'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        layer_input = input[0].detach()
        module.register_exts_buffer('layer_input', layer_input)
        module.output_size = tuple(output.size())

    # --- end of hooks ---

    # --- Hessian-vector product with the input Hessian ---
    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Return CVP with respect to the input."""

        def _input_hessian_vp(v):
            """Multiplication by the Hessian w.r.t. the input."""
            return self._input_jacobian_transpose(
                output_hessian(self._input_jacobian(v)))

        return _input_hessian_vp

    def _input_jacobian(self, v):
        """Apply the Jacobian with respect to the input."""
        batch, in_channels, in_x, in_y = tuple(self.layer_input.size())
        assert tuple(v.size()) == (self.layer_input.numel(), )
        result = v.view(batch, in_channels, in_x, in_y)
        result = functional.conv2d(
            result,
            self.weight.data,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        assert tuple(result.size()) == self.output_size
        return result.view(-1)

    def _input_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the input."""
        batch, in_channels, in_x, in_y = tuple(self.layer_input.size())
        assert tuple(v.size()) == (prod(self.output_size), )
        result = v.view(*self.output_size)
        result = functional.conv_transpose2d(
            result,
            self.weight.data,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        assert tuple(result.size()) == tuple(self.layer_input.size())
        return result.view(-1)

    # --- Hessian-vector products with the parameter Hessians ---
    # override
    def parameter_hessian(self, output_hessian):
        """Initialize VPs with the layer parameter Hessian."""
        if self.bias is not None:
            self.init_bias_hessian(output_hessian)
        self.init_weight_hessian(output_hessian)

    # --- bias term ---
    def init_bias_hessian(self, output_hessian):
        """Initialize bias Hessian-vector product."""

        def _bias_hessian_vp(v):
            """Multiplication by the bias Hessian."""
            return self._bias_jacobian_transpose(
                output_hessian(self._bias_jacobian(v)))

        self.bias.hvp = _bias_hessian_vp

    def _bias_jacobian(self, v):
        """Apply the Jacobian with respect to the bias."""
        assert tuple(v.size()) == (self.bias.numel(), )
        result = v.view(1, self.bias.numel(), 1, 1)
        result = result.expand(*self.output_size)
        assert tuple(result.size()) == self.output_size
        return result.contiguous().view(-1)

    def _bias_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the bias."""
        assert tuple(v.size()) == (prod(self.output_size), )
        result = v.view(*self.output_size).sum(3).sum(2).sum(0)
        assert tuple(result.size()) == (self.bias.numel(), )
        return result

    def init_weight_hessian(self, output_hessian):
        """Initialize weight Hessian-vector product."""

        def _weight_hessian_vp(v):
            """Multiplication by the weight Hessian."""
            return self._weight_jacobian_transpose(
                output_hessian(self._weight_jacobian(v)))

        self.weight.hvp = _weight_hessian_vp

    def _weight_jacobian(self, v):
        """Apply the Jacobian with respect to the weights."""
        batch, out_channels, _, _ = self.output_size
        assert tuple(v.size()) == (self.weight.numel(), )
        result = v.view(1, out_channels, -1)
        result = result.expand(batch, out_channels, -1)
        result = einsum('bij,bki->bkj', (functional.unfold(
            self.layer_input,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride), result))
        result = result.contiguous().view(-1)
        assert tuple(result.size()) == (prod(self.output_size), )
        return result

    def _weight_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the weights."""
        batch, out_channels, _, _ = self.output_size
        assert tuple(v.size()) == (prod(self.output_size), )
        result = v.view(batch, out_channels, -1)
        result = einsum('bij,bkj->ki', (functional.unfold(
            self.layer_input,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride), result))
        result = result.contiguous().view(-1)
        assert tuple(result.size()) == (self.weight.numel(), )
        return result

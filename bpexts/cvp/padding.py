"""Curvature-vector products for padding layers."""

from numpy import prod
from torch import (arange, zeros, tensor)
from torch.nn import functional
from torch.nn import ZeroPad2d
from ..hbp.module import hbp_decorate
from ..utils import einsum


class CVPZeroPad2d(hbp_decorate(ZeroPad2d)):
    """2D Zero padding with recursive Hessian-vector products."""
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, ZeroPad2d):
            raise ValueError("Expecting torch.nn.ZeroPad2s, got {}".format(
                torch_layer.__class__))
        # create instance
        return cls(torch_layer.padding)

    # override
    def hbp_hooks(self):
        """Install hook storing unfolded input."""
        self.register_exts_forward_hook(self.store_input_and_output_dimensions)

    # --- hooks ---
    @staticmethod
    def store_input_and_output_dimensions(module, input, output):
        """Save input and dimensions of the output to the layer.

        Intended use as forward hook.
        Initialize module buffer 'input_size' and attribute
        'output_size'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        module.input_size = tuple(input[0].size())
        module.output_size = tuple(output.size())

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
        assert tuple(v.size()) == (prod(self.input_size), )
        result = v.view(*self.input_size)
        result = functional.pad(result, self.padding, 'constant', self.value)
        assert tuple(result.size()) == self.output_size
        return result.contiguous().view(-1)

    def _input_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the input."""
        assert tuple(v.size()) == (prod(self.output_size), )
        pad_left, pad_right, pad_top, pad_bottom = self.padding
        result = v.view(*self.output_size)
        result = result[:, :, pad_top:-pad_bottom, pad_left:-pad_right]
        assert tuple(result.size()) == self.input_size
        return result.contiguous().view(-1)

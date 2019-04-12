"""Curvature-vector products for maxpool2d layer."""

import torch
from torch import Tensor
from torch.nn import MaxPool2d, functional
from ..hbp.module import hbp_decorate
from ..utils import set_seeds


class CVPMaxPool2d(hbp_decorate(MaxPool2d)):
    """2d Max pooling with recursive curvature-vector products.

    Note:
    -----
    For convenience, the module does not return the indices after
    the forward pass. Instead, they can be accessed by the module
    attribute ``pool_indices``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override
    def hbp_hooks(self):
        """Pooling indices and in/out dimensions are saved in forward."""
        pass

    def forward(self, x):
        """Return only the pooled tensor, but save indices as buffer.

        Initialize module buffer ``self.pool_indices``.
        """
        self.return_indices = True
        out, idx = super().forward(x)
        # save quantities
        self.input_shape = tuple(x.size())
        self.output_shape = tuple(out.size())
        self.register_exts_buffer("pool_indices", idx)
        self.register_exts_buffer("pool_indices_1d",
                                  self._convert_pooling_indices(idx))
        return out

    def _convert_pooling_indices(self, idx):
        """Convert the pooling indices returned from the forward pass into
        the one-dimensional index scheme."""
        batch, channels, in_x, in_y = self.input_shape
        # convert values pool_indices to one-dimensional indices
        idx_batch_offset = (channels * in_x * in_y * torch.arange(batch)).to(
            idx.device).view(batch, 1, 1, 1)
        idx_channel_offset = (in_x * in_y * torch.arange(channels)).to(
            idx.device).view(1, channels, 1, 1)
        idx_1d = idx + idx_batch_offset.expand(
            *self.output_shape) + idx_channel_offset.expand(*self.output_shape)
        return idx_1d.view(-1)

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
        """Apply the Jacobian with respect to the input.

        Basically, one has to retrieve the elements in ``v`` according
        to the max-pooled indices.

        Reference:
        ----------
        https://discuss.pytorch.org/t/maxpool2d-indexing-order/8281
        """
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        assert tuple(v.size()) == (batch * channels * in_x * in_y, )
        result = v.view(batch, channels, -1)
        result = result.gather(2, self.pool_indices.view(batch, channels, -1))
        assert tuple(result.size()) == (batch, channels, out_x * out_y)
        return result.view(-1)

    def _input_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the input."""
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        assert tuple(v.size()) == (batch * channels * out_x * out_y, )
        # accumulate values in v according to the 1d indices
        result = torch.zeros(batch * channels * in_x * in_y, device=v.device)
        result.index_put_((self.pool_indices_1d, ), v, accumulate=True)
        return result

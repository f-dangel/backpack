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
        self.input_shape = tuple(x.size())
        self.return_indices = True
        out, idx = super().forward(x)
        self.output_shape = tuple(out.size())
        self.register_exts_buffer("pool_indices", idx)
        return out

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
        """Apply the transposed Jacobian with respect to the input.

        Unpool according to the pooling indices.
        """
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        assert tuple(v.size()) == (batch * channels * out_x * out_y, )
        """
        result = v.view(*self.output_shape)
        result = functional.max_unpool2d(
            result,
            self.pool_indices,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=self.input_shape)
        assert tuple(result.size()) == self.input_shape
        """
        idx = self.pool_indices.clone()
        for b in range(batch):
            idx[b, :, :, :] += b * channels * in_x * in_y
            for i in range(channels):
                idx[b, i, :, :] += i * in_x * in_y
        idx = idx.view(-1)
        result = torch.zeros(batch * channels * in_x * in_y).to(v.device)
        result.index_put_((idx, ), v, accumulate=True)
        return result

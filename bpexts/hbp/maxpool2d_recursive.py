"""Hessian backpropagation for 2d max pooling."""

import collections
from torch import Tensor, zeros
from .maxpool2d import HBPMaxPool2d


class HBPMaxPool2dRecursive(HBPMaxPool2d):
    """2d Max pooling with recursive Hessian-vector products."""

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Return matrix-vector multiplication routine with input Hessian."""
        if isinstance(output_hessian, Tensor):
            output_hessian_vp = output_hessian.detach().matmul
        elif isinstance(output_hessian, collections.Callable):
            output_hessian_vp = output_hessian
        else:
            raise ValueError(
                "Expecting torch.Tensor or function, but got\n{}".format(
                    output_hessian))
        return self.input_hessian_vp(output_hessian_vp)

    def input_hessian_vp(self, output_hessian_vp):
        """Return matrix-vector multiplication routine with the input Hessian.

        Parameters:
        -----------
        output_hessian_vp : function
            Matrix-vector multiplication routine with the output Hessian.

        Returns:
        --------
        function
            Matrix-vector multiplication routine with the input Hessian.
        """
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape

        def _input_hessian_vp(v):
            """Multiplication by the Hessian w.r.t. input."""
            # apply batch-averaged Jacobian
            result = self._mean_input_jacobian(v)
            # apply output Hessian
            result = output_hessian_vp(result)
            # apply batch-averaged transpose Jacobian
            result = self._mean_input_jacobian_transpose(result)
            return result

        return _input_hessian_vp

    def _mean_input_jacobian(self, v):
        """Apply the batch-averaged Jacobian with respect to the input."""
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        assert tuple(v.size()) == (channels * in_x * in_y, )
        result = v.view(1, channels, -1)
        result = result.expand(batch, channels, -1)
        result = result.gather(2, self.pool_indices.view(batch, channels, -1))
        assert tuple(result.size()) == (batch, channels, out_x * out_y)
        return result.mean(0).view(-1)

    def _mean_input_jacobian_transpose(self, v):
        """Apply the batch-averaged tranpose Jacobian w.r.t. the input."""
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        assert tuple(v.size()) == (channels * out_x * out_y, )
        result = zeros(batch, channels, in_x * in_y, device=v.device)
        result.scatter_add_(
            2, self.pool_indices.view(batch, channels, -1),
            v.view(1, channels, -1).expand(batch, channels, -1))
        assert tuple(result.size()) == (
            batch,
            channels,
            in_x * in_y,
        )
        return result.mean(0).view(-1)

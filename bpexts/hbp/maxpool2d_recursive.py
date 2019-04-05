"""Hessian backpropagation for 2d max pooling."""

from torch import Tensor
from torch.nn import MaxPool2d
from torch.nn.functional import max_unpool2d
from .module import hbp_decorate
from ..utils import set_seeds


class HBPMaxPool2dRecursive(hbp_decorate(MaxPool2d)):
    """2d Max pooling with recursive Hessian-vector products.

    Note:
    -----
    For convenience, the module does not return the indices after
    the forward pass. Instead, they can be accessed by the module
    attribute ``pool_indices``.
    """

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
        x, idx = super().forward(x)
        self.output_shape = tuple(x.size())
        self.register_exts_buffer("pool_indices", idx)
        return x

    def unpool2d(self, x):
        """Unpool according to the indices stored in the last forward pass."""
        return max_unpool2d(
            x,
            self.pool_indices,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding)

    # override
    def input_hessian(self,
                      output_hessian,
                      compute_input_hessian=True,
                      modify_2nd_order_terms='none'):
        """Return matrix-vector multiplication routine with input Hessian."""
        if compute_input_hessian is False:
            return None
        else:
            if compute_input_hessian is False:
                return None
            output_hessian_vp = None
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
            assert tuple(v.size()) == (channels * in_x * in_y, )
            # apply batch-averaged Jacobian
            result = v.view(1, channels, in_x, in_y)
            # clone for each batch sample
            result = result.view(1, channels, in_x, in_y)
            result = result.expand(batch, channels, in_x, in_y)
            # indices in flattened input
            idx = self.pool_indices.clone()
            assert tuple(idx.size()) == (
                batch,
                channels,
                out_x,
                out_y,
            )
            for b in range(batch):
                idx[b, :, :, :] += b * channels * in_x * in_y
                for i in range(channels):
                    idx[b, i, :, :] += i * in_x * in_y
            idx = idx.view(-1)
            result = result.contiguous().view(-1)
            result = result[idx]
            # batch averaged Jacobian
            result = result.view(batch, -1).mean(0)

            assert tuple(result.size()) == (channels * out_x * out_y, )
            # apply the output Hessian
            result = output_hessian_vp(result)
            assert tuple(result.size()) == (channels * out_x * out_y, )
            result = result.view(channels, out_x, out_y)
            # clone for each batch sample
            result = result.expand(batch, channels, out_x, out_y)
            assert tuple(result.size()) == (
                batch,
                channels,
                out_x,
                out_y,
            )
            # Jacobian for each sample
            print(result.size())
            print(self.pool_indices)
            result = self.unpool2d(result)
            assert tuple(result.size()) == (
                batch,
                channels,
                in_x,
                in_y,
            )
            # average over batch dimension
            result = result.mean(0)
            result = result.view(-1)
            return result

        return _input_hessian_vp

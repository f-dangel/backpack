"""Hessian backpropagation for 2d max pooling."""

from torch import Tensor, arange, zeros, zeros_like
from torch.nn import MaxPool2d
from torch.nn.functional import max_unpool2d
from .module import hbp_decorate
from ..utils import set_seeds


class HBPMaxPool2d(hbp_decorate(MaxPool2d)):
    """2d Max pooling with HBP.

    Note:
    -----
    For convenience, the module does not return the indices after
    the forward pass. Instead, they can be accessed by the module
    attribute ``pool_indices``.
    """
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, MaxPool2d):
            raise ValueError("Expecting torch.nn.MaxPool2d, got {}".format(
                torch_layer.__class__))
        # create instance
        maxpool2d = cls(
            torch_layer.kernel_size,
            stride=torch_layer.stride,
            padding=torch_layer.padding,
            dilation=torch_layer.dilation)
        return maxpool2d

    # override
    def set_hbp_approximation(self,
                              average_input_jacobian=False,
                              average_parameter_jacobian=None):
        """No approximation for parameter Hessian required."""
        super().set_hbp_approximation(
            average_input_jacobian=average_input_jacobian,
            average_parameter_jacobian=None)

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
        return out

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Return Hessian with respect to the input."""
        # shape information
        batch, channels, in_x, in_y = self.input_shape
        _, _, out_x, out_y = self.output_shape
        # indices in flattened input
        offset = (arange(
            channels,
            dtype=self.pool_indices.dtype,
            device=self.pool_indices.device) * (in_x * in_y)).view(-1, 1, 1)
        offset = offset.expand(-1, out_x, out_y)
        # average over batch
        if self.average_input_jac is False:
            # Hessian
            h_in = zeros(
                channels * in_x * in_y,
                channels * in_x * in_y,
                device=output_hessian.device)
            for b in range(self.pool_indices.size(0)):
                idx_map = self.pool_indices[b, :] + offset
                idx_map = idx_map.view(-1)
                # TODO: Express more efficiently
                # sum rows
                temp = zeros(
                    channels * in_x * in_y,
                    channels * out_x * out_y,
                    device=output_hessian.device)
                for n, idx in enumerate(idx_map):
                    temp[idx, :] += output_hessian[n, :]
                # sum columns
                temp2 = zeros(
                    channels * in_x * in_y,
                    channels * in_x * in_y,
                    device=output_hessian.device)
                for n, idx in enumerate(idx_map):
                    temp2[:, idx] += temp[:, n]
                h_in += temp2
            return h_in / batch
        elif self.average_input_jac is True:
            temp = zeros(
                channels * in_x * in_y,
                channels * out_x * out_y,
                device=output_hessian.device)
            # average over lines
            for b in range(self.pool_indices.size(0)):
                idx_map = self.pool_indices[b, :] + offset
                idx_map = idx_map.view(-1)
                for n, idx in enumerate(idx_map):
                    temp[idx, :] += output_hessian[n, :]
            h_in = zeros(
                channels * in_x * in_y,
                channels * in_x * in_y,
                device=output_hessian.device)
            # average over columns
            for b in range(self.pool_indices.size(0)):
                idx_map = self.pool_indices[b, :] + offset
                idx_map = idx_map.view(-1)
                for n, idx in enumerate(idx_map):
                    h_in[:, idx] += temp[:, n]
            return h_in / batch**2
        else:
            raise ValueError('Unknown value for average_input_jac : {}'.format(
                self.average_input_jac))

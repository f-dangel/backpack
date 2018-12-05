"""Hessian backpropagation for 2D convolution."""

from numpy import prod
from torch import (eye, einsum)
from torch.nn import functional
from torch.nn import Conv2d
from .module import hbp_decorate


class HBPConv2d(hbp_decorate(Conv2d)):
    """2D Convolution with Hessian backpropagation functionality."""
    # override
    def hbp_hooks(self):
        """Install hook storing unfolded input."""
        self.register_exts_forward_pre_hook(self.store_unfolded_input)

    def unfold(self, input):
        """Unfold input using convolution hyperparameters."""
        return functional.unfold(input,
                                 kernel_size=self.kernel_size,
                                 dilation=self.dilation,
                                 padding=self.padding,
                                 stride=self.stride)

    # --- hooks ---
    @staticmethod
    def store_unfolded_input(module, input):
        """Save unfolded input.

        Indended use as pre-forward hook.
        Initialize module buffer 'unfolded_input'
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        unfolded_input = module.unfold(input[0]).detach()
        module.register_exts_buffer('unfolded_input', unfolded_input)
    # --- end of hooks ---

    # override
    def parameter_hessian(self, output_hessian):
        """Compute parameter Hessian.

        The Hessian of the bias (if existent) is stored in the attribute
        self.bias.hessian. Hessian-vector product function is stored in
        self.bias.hvp.

        The Hessian of the weight is not computed explicitely for memory
        efficiency. Instead, a method is stored in self.weight.hessian,
        that produces the explicit Hessian matrix when called. Hessian-
        vector product function is stored in self.weight.hvp.
        """
        if self.bias is not None:
            self.init_bias_hessian(output_hessian.detach())
        self.init_weight_hessian(output_hessian.detach())

    # override
    def input_hessian(self, output_hessian, compute_input_hessian=True):
        """Compute the Hessian with respect to the layer input."""
        if compute_input_hessian is False:
            return None
        else:
            return self.unfolded_input_hessian(output_hessian.detach())

    def unfolded_input_hessian(self, out_h):
        """Compute Hessian with respect to the layer's unfolded input."""
        kernel_matrix = self.weight.view(self.out_channels, -1)
        num_patches = self.unfolded_input.size()[2]
        id_num_patches = eye(num_patches)
        h_tensor_structure = 2 * (self.out_channels, num_patches)
        unfolded_hessian = einsum('ij,kl,ilmn,mp,no->jkpo',
                                  (kernel_matrix,
                                   id_num_patches,
                                   out_h.view(h_tensor_structure),
                                   kernel_matrix,
                                   id_num_patches))
        final_shape = 2 * (prod(self.unfolded_input.size()[1:]),)
        return unfolded_hessian.view(final_shape)

    def init_bias_hessian(self, output_hessian):
        """Initialized bias attributes hessian and hvp.

        Initializes:
        ------------
        self.bias.hessian: Holds a matrix representing the batch-averaged
                           Hessian with respect to the bias
        self.bias.hvp: Provides implicit matrix-vector multiplication
                       routine by the batch-averaged bias Hessian

        Parameters:
        -----------
        out_h (torch.Tensor): Batch-averaged Hessian with respect to
                              the layer's outputs
        """
        patch = prod(self.kernel_size)
        shape = (self.out_channels, patch, self.out_channels, patch)
        self.bias.hessian = output_hessian.view(shape).sum(3).sum(1)
        self.bias.hvp = self._bias_hessian_vp

    def init_weight_hessian(self, out_h):
        """Create weight attributes hessian and hvp.

        Initializes:
        ------------
        self.weight.hessian: Holds a function which, when called, returns
                             a matrix representing the batch-averaged
                             Hessian with respect to the weights
        self.weight.hvp: Provides implicit matrix-vector multiplication
                         routine by the batch-averaged weight Hessian

        Parameters:
        -----------
        out_h (torch.Tensor): Batch-averaged Hessian with respect to
                              the layer's outputs
        """
        self.weight.hessian = self._compute_weight_hessian(out_h)
        self.weight.hvp = self._weight_hessian_vp(out_h)

    def _bias_hessian_vp(self, v):
        """Matrix multiplication by bias Hessian.

        Parameters:
        -----------
        v (torch.Tensor): Vector which is to be multiplied by the Hessian

        Returns:
        --------
        result (torch.Tensor): bias_hessian * v
        """
        return self.bias.hessian.matmul(v)

    def _weight_hessian_vp(self, out_h):
        """Matrix multiplication by weight Hessian.
        """
        def hvp(v):
            """Matrix-vector product with weight Hessian.

            Use approximation
            weight_hessian = (I \otimes X) output_hessian (I \otimes X^T).

            Parameters:
            -----------
            v (torch.Tensor): Vector which is multiplied by the Hessian
           """
            if not len(v.size()) == 1:
                raise ValueError('Require one-dimensional tensor')
            batch = self.unfolded_input.size()[0]
            id_out_channels = eye(self.out_channels)
            # reshape vector into (out_channels, -1)
            temp = v.view(self.out_channels, -1)
            # perform tensor network contraction
            result = einsum('ij,bkl,jlmp,mn,bop,no->ik',
                            (id_out_channels,
                             self.unfolded_input,
                             out_h.view(self._out_h_tensor_structure()),
                             id_out_channels,
                             self.unfolded_input,
                             temp)) / batch
            return result.view(v.size())
        return hvp

    def _compute_weight_hessian(self, out_h):
        """Compute weight Hessian from output Hessian.

        Use approximation
        weight_hessian = (I \otimes X) output_hessian (I \otimes X^T).
        """
        def weight_hessian():
            """Compute matrix form of the weight Hessian when called."""
            batch = self.unfolded_input.size()[0]
            id_out_channels = eye(self.out_channels)
            # compute the weight Hessian
            w_hessian = einsum('ij,bkl,jlmp,mn,bop->ikno',
                               (id_out_channels,
                                self.unfolded_input,
                                out_h.view(self._out_h_tensor_structure()),
                                id_out_channels,
                                self.unfolded_input)) / batch
            # reshape into square matrix
            num_weight = self.weight.numel()
            return w_hessian.view(num_weight, num_weight)
        return weight_hessian

    def _out_h_tensor_structure(self):
        """Return tensor shape of output Hessian for further processing.

        The rank-4 shape is given by (out_channels, patch_size, out_channels,
        patch_size)."""
        patch_size = prod(self.weight.size()[2:])
        return 2 * (self.out_channels, patch_size)

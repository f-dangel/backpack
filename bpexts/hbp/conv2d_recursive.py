"""Recursive Hessian-vector products in spirit of HBP for 2d Convolution."""

import collections
from .conv2d import HBPConv2d
from torch import Tensor, einsum


class HBPConv2dRecursive(HBPConv2d):
    """2d Convolution with recursive Hessian-vector products."""

    # override
    def parameter_hessian(self, output_hessian):
        """Define Hessian-vector products with the parameters.


        vector product function is stored in self.weight.hvp.
        """
        # use matrix-vector routine if receiving a tensor
        output_hessian_vp = None
        if isinstance(output_hessian, Tensor):
            output_hessian_vp = output_hessian.detach().matmul
        elif isinstance(output_hessian, collections.Callable):
            output_hessian_vp = output_hessian
        else:
            raise ValueError(
                "Expecting torch.Tensor or function, but got\n{}".format(
                    output_hessian))
        if self.bias is not None:
            self.init_bias_hessian_vp(output_hessian_vp)
        self.init_weight_hessian(output_hessian_vp)

    def init_bias_hessian_vp(self, output_hessian_vp):
        """Initialize bias hvp.

        Initializes:
        ------------
        self.bias.hvp: Provides implicit matrix-vector multiplication
                       routine by the batch-averaged bias Hessian

        Parameters:
        -----------
        output_hessian_vp (function): Matrix-vector multiplication routine
                                      with the output Hessian.
        """
        out_channels, num_patches, _, _ = self.h_out_tensor_structure()

        def bias_hessian_vp(v):
            """Multiply with the approximate bias Hessian.

            Apply the equation
                (Hb) v = [D^T (H_out) D] v,
            where Hb, H_out, and D denote the bias Hessian, the output
            Hessian, and the Jacobian, respectively

            Parameters:
            -----------
            v : 1d torch.Tensor
                One dimensional vector with the same number of elements
                as the bias term.

            Returns:
            --------
            Vector v multiplied with the approximated bias Hessian
            """
            assert tuple(v.size()) == (out_channels, )
            v = v.view(-1, 1)
            # apply the Jacobian
            result = v.expand(out_channels, num_patches).contiguous().view(-1)
            assert tuple(result.size()) == (out_channels * num_patches, )
            # apply the Hessian with respect to the outputs
            result = output_hessian_vp(result)
            assert tuple(result.size()) == (out_channels * num_patches, )
            # apply the transposed Jacobian
            result = result.view(out_channels, num_patches).sum(1)
            assert tuple(result.size()) == (out_channels, )
            return result

        self.bias.hvp = bias_hessian_vp

    def init_weight_hessian(self, output_hessian_vp):
        """Initialize weight Hessian HVP.

        Initializes:
        ------------
        self.weight.hvp: Provides implicit matrix-vector multiplication
                         routine by the batch-averaged weight Hessian

        Parameters:
        -----------
        output_hessian_vp (function): Matrix-vector multiplication routine
                                      with the output Hessian.
        """
        out_channels, num_patches, _, _ = self.h_out_tensor_structure()

        def weight_hessian_vp(v):
            """Matrix multiplication by approximate weight Hessian.

            HW = (I \otimes X) H_out (I \otimes X^T).

            Parameters:
            -----------
            v : 1d torch.Tensor
                Vector which is multiplied by the Hessian

            Returns:
            --------
            1d torch.Tensor
                Result of the computation HW v
            """
            assert tuple(v.size()) == (self.weight.numel(), )
            # apply the Jacobian
            result = v.view(out_channels, -1)
            result = einsum('ij,ki->kj', (self.mean_unfolded_input, result))
            assert tuple(result.size()) == (out_channels, num_patches)
            result = result.contiguous().view(-1)
            # apply the output Hessian
            result = output_hessian_vp(result)
            assert tuple(result.size()) == (out_channels * num_patches, )
            # apply the Jacobian
            result = result.view(out_channels, -1)
            result = einsum('ij,kj->ki', (self.mean_unfolded_input, result))

            assert tuple(
                result.size()) == (out_channels,
                                   self.weight.numel() // out_channels)
            result = result.contiguous().view(-1)
            assert tuple(result.size()) == (self.weight.numel(), )
            return result

        self.weight.hvp = weight_hessian_vp

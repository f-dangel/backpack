"""Hessian backpropagation for a linear layer with Kronecker-factored
weight Hessian approximation."""

from torch import einsum
from .linear import HBPLinear


class HBPLinear2(HBPLinear):
    """Use weight Hessian approximation of KFRA paper."""

    # override
    def hbp_hooks(self):
        """Install hooks required for Hessian backward pass.

        The computation of the Hessian usually involves quantities that
        need to be computed during a forward or backward pass.
        """
        self.register_exts_forward_pre_hook(self.store_mean_input_outer)

    # --- hooks ---
    @staticmethod
    def store_mean_input_outer(module, input):
        """Save mean(input * input^T) of layer input.

        Intended use as pre-forward hook.
        Initialize module buffer 'mean_input_outer'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        batch = input[0].size()[0]
        mean_input_outer = einsum(
            'bi,bj->ij', (input[0].detach(), input[0].detach())) / batch
        module.register_exts_buffer('mean_input_outer', mean_input_outer)

    def weight_hessian(self, out_h):
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

        def hvp(v):
            r"""Matrix-vector product with weight Hessian.

            Use approximation
             weight_hessian = output_hessian \otimes
                              mean(input) \otimes mean(input^T)

            Parameters:
            -----------
            v (torch.Tensor): Vector which is multiplied by the Hessian
           """
            if not len(v.size()) == 1:
                raise ValueError('Require one-dimensional tensor')
            num_outputs = out_h.size()[0]
            num_inputs = self.mean_input_outer.size()[1]
            result = v.reshape(num_outputs, num_inputs)
            # order matters for memory consumption:
            #   - mean_input has shape (num_inputs, 1)
            #   - out_h has shape (num_outputs, num_outputs)
            # assume num_outputs is smaller than num_inputs
            result = out_h.matmul(result)
            result = result.matmul(self.mean_input_outer)
            return result.reshape(v.size())

        self.weight.hvp = hvp

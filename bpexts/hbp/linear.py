"""Hessian backpropagation for a linear layer."""

from torch import einsum
from torch.nn import Linear
from .module import hbp_decorate


class HBPLinear(hbp_decorate(Linear)):
    """Linear layer with Hessian backpropagation functionality.

    Applies x A^T + b to a given input x. A is the weight matrix,
    b the bias term.

    More details on the structure of the Hessian of a linear layer
    can be found for instance in Chen et al: BDA-PCH (2018).

    Parameters:
    -----------
    in_features (int): Number of input features
    out_features (int): Number of output features
    bias (bool): Use a bias term

    Details:
    --------
    For a linear layer (line stacking vectorization convention),
    i) the bias Hessian is equivalent to the Hessian with respect
       to the linear layer's output (output_hessian)
    ii) the weight Hessian can be obtained from the output Hessian
        by means of the layer's input and the relation
        weight_hessian = output_hessian \otimes input \otimes input^T
    iii) the Hessian with respect to the inputs is given by
         input_hessian = weight^T * output_hessian * weight.
    """

    # override
    def hbp_hooks(self):
        """Install hooks required for Hessian backward pass.

        The computation of the Hessian usually involves quantities that
        need to be computed during a forward or backward pass.
        """
        # more accurate approximation
        # self.register_exts_forward_pre_hook(self.store_input_mean_outer)
        # rough but more efficient
        self.register_exts_forward_pre_hook(self.store_mean_input)

    # --- hooks ---
    @staticmethod
    def store_mean_input(module, input):
        """Save batch average of input of layer.

        Intended use as pre-forward hook.
        Initialize module buffer 'mean_input'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        mean_input = input[0].detach().mean(0).unsqueeze_(0)
        module.register_exts_buffer('mean_input', mean_input)

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

        i) the bias Hessian is equivalent to the Hessian with respect
           to the linear layer's output (output_hessian)
        ii) the weight Hessian can be obtained from the output Hessian
            by means of the layer's input and the relation
            weight_hessian = output_hessian \otimes input \otimes input^T
        """
        if self.bias is not None:
            self.init_bias_hessian(output_hessian.detach())
        self.init_weight_hessian(output_hessian.detach())

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Compute the Hessian with respect to the layer input.

        Exploited relation:
         input_hessian = weight * output_hessian * weight^T.
        """
        return self.weight.t().matmul(output_hessian).matmul(self.weight)

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
        self.bias.hessian = output_hessian
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
             weight_hessian = output_hessian \otimes
                              mean(input) \otimes mean(input^T)

            Parameters:
            -----------
            v (torch.Tensor): Vector which is multiplied by the Hessian
           """
            if not len(v.size()) == 1:
                raise ValueError('Require one-dimensional tensor')
            num_outputs = out_h.size()[0]
            num_inputs = self.mean_input.size()[1]
            result = v.reshape(num_outputs, num_inputs)
            # order matters for memory consumption:
            #   - mean_input has shape (num_inputs, 1)
            #   - out_h has shape (num_outputs, num_outputs)
            # assume num_outputs is smaller than num_inputs
            temp = out_h.matmul(result)
            temp = temp.matmul(self.mean_input.t())
            return temp.matmul(self.mean_input).reshape(v.size())

        return hvp

    def _compute_weight_hessian(self, output_hessian):
        """Compute weight Hessian from output Hessian.

        Use approximation
        weight_hessian = output_hessian \otimes mean(input \otimes input^T).
        """

        def weight_hessian():
            """Compute matrix form of the weight Hessian when called.
            """
            w_hessian = einsum('ij,k,l->ikjl',
                               (output_hessian, self.mean_input.squeeze(),
                                self.mean_input.squeeze()))
            dim_i, dim_k, _, _ = w_hessian.size()
            dim = dim_i * dim_k
            return w_hessian.reshape(dim, dim)

        return weight_hessian

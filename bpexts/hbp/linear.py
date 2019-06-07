"""Hessian backpropagation for a linear layer."""

from warnings import warn
from torch import einsum, Tensor
from torch.nn import Linear
from .module import hbp_decorate


class HBPLinear(hbp_decorate(Linear)):
    r"""Linear layer with Hessian backpropagation functionality.

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
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Linear):
            raise ValueError("Expecting torch.nn.Linear, got {}".format(
                torch_layer.__class__))
        # create instance
        linear = cls(
            in_features=torch_layer.in_features,
            out_features=torch_layer.out_features,
            bias=torch_layer.bias is not None)
        # copy parameters
        linear.weight = torch_layer.weight
        linear.bias = torch_layer.bias
        return linear

    # override
    def set_hbp_approximation(self,
                              average_input_jacobian=None,
                              average_parameter_jacobian=True):
        """Not sure if useful to implement"""
        if average_input_jacobian is not None:
            warn(
                'HBPLinear: You tried to set the input Hessian approximation',
                ' to {}, but both approximations yield the same behavior.'.
                format(average_input_jacobian), 'Resetting to None.')
        super().set_hbp_approximation(
            average_input_jacobian=None,
            average_parameter_jacobian=average_parameter_jacobian)

    # override
    def hbp_hooks(self):
        """Install hooks required for Hessian backward pass.

        The computation of the Hessian usually involves quantities that
        need to be computed during a forward or backward pass.
        """
        self.register_exts_forward_pre_hook(self.store_input)

    # override
    def compute_backward_hessian_quantities(self):
        """Compute quantities for weight Hessian depending on approximation."""
        if self.average_param_jac == True:
            self.register_exts_buffer('mean_input', self._mean_input())
        elif self.average_param_jac == False:
            self.register_exts_buffer('input_kron_mean',
                                      self._input_kron_mean())
        else:
            raise ValueError('Unknown value for average_param_jac : {}'.format(
                self.average_param_jac))
        # free input
        self.input = None

    def _mean_input(self):
        """Compute batch average of module input. Return flat."""
        return self.input.mean(0).detach().view(-1)

    def _input_kron_mean(self):
        """Compute batch average of input outer product."""
        batch = self.input.size(0)
        return einsum('bi,bj->ij', (self.input.view(
            batch, -1), self.input.view(batch, -1))).detach() / batch

    # override
    def parameter_hessian(self, output_hessian):
        r"""Compute parameter Hessian.

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
            self.bias_hessian(output_hessian.detach())
        self.weight_hessian(output_hessian.detach())

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Compute the Hessian with respect to the layer input.

        Exploited relation:
         input_hessian = weight * output_hessian * weight^T.
        """
        assert isinstance(output_hessian, Tensor)
        in_hessian = self.weight.t().matmul(output_hessian).matmul(self.weight)
        return in_hessian.detach()

    def bias_hessian(self, output_hessian):
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
        self.bias.hvp = output_hessian.matmul

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
            assert tuple(v.size()) == (self.weight.numel(), )
            if self.average_param_jac == True:
                result = einsum(
                    'ij,jk,k,l->il',
                    (out_h, v.view(self.out_features, self.in_features),
                     self.mean_input, self.mean_input))
            elif self.average_param_jac == False:
                result = einsum(
                    'ij,jk,kl->il',
                    (out_h, v.view(self.out_features, self.in_features),
                     self.input_kron_mean))
            else:
                raise ValueError(
                    'Unknown value for average_param_jac : {}'.format(
                        self.average_param_jac))
            result = result.view(-1)
            assert tuple(v.size()) == (self.weight.numel(), )
            return result

        self.weight.hvp = hvp

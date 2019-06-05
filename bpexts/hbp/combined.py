"""Hessian backpropagation for a composition of activation and linear layer.

This composition leads to the scheme from BDA-PCH and KFRA.
"""

from torch import einsum
from torch.nn import Module
from .linear import HBPLinear
from .sigmoid import HBPSigmoid
from .module import hbp_decorate


class HBPCompositionActivationLinear(hbp_decorate(Module)):
    """linear(activation) layer with HBP in BDA-PCH style.

    Applies phi(x) A^T + b.

    In the backpropagation procedure, this yields the Hessian
    approximations from the KFRA and BDA-PCH paper.

    More details on the structure of the Hessian of a linear layer
    can be found for instance in Chen et al: BDA-PCH (2018).

    Parameters:
    -----------
    nonlinear_hbp_cls (class): HBP class of an elementwise activation
    in_features (int): Number of input features
    out_features (int): Number of output features
    bias (bool): Use a bias term
    """

    def __init__(self, nonlinear_hbp_cls, in_features, out_features,
                 bias=True):
        super().__init__()
        self.activation = nonlinear_hbp_cls()
        self.linear = HBPLinear(
            in_features=in_features, out_features=out_features, bias=bias)

    # override
    def set_hbp_approximation(self,
                              average_input_jacobian=True,
                              average_parameter_jacobian=True):
        """Not sure if useful to implement"""
        if average_input_jacobian is not True:
            raise NotImplementedError
        if average_parameter_jacobian is not True:
            raise NotImplementedError
        super().set_hbp_approximation(
            average_input_jacobian=average_input_jacobian,
            average_parameter_jacobian=average_parameter_jacobian)

    def enable_hbp(self):
        super().enable_hbp()
        try:
            self.activation.enable_hbp()
            self.linear.enable_hbp()
        except AttributeError:
            pass

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def forward(self, input):
        """Apply activation, followed by an affine transformation.

        Note:
        -----
        The application of the Sigmoid is currently happening twice (also
        in a pre-forward hook which stores the quantities required for Hessian
        backpropagation). This is conceptionally cleaner but obviously not
        the most efficient way of doing so.
        """
        return self.linear(self.activation(input))

    # override
    def parameter_hessian(self, output_hessian):
        """Compute parameter Hessians."""
        self.linear.parameter_hessian(output_hessian)

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Compute the Hessian with respect to the layer input.

        Exploited relation: recursion in BDA-PCH paper.
        """
        in_hessian = self._compute_gauss_newton(output_hessian)
        idx = list(range(in_hessian.size()[0]))
        in_hessian[idx, idx] = in_hessian[idx, idx] +\
            self._compute_residuum(modify_2nd_order_terms)
        return in_hessian

    def _compute_gauss_newton(self, output_hessian):
        """Compute the Gauss-Newton matrix from the output Hessian.

        The generalized Gauss-Newton matrix (ggn) is given by:
            ggn = J^T * H_out * J,
        where J denotes the Jacobian.
        """
        jacobian_t = self._compute_jacobian_t()
        ggn = jacobian_t.matmul(output_hessian).matmul(jacobian_t.t())
        return ggn.detach()

    def _compute_jacobian_t(self):
        """Compute the transpose Jacobian mean(d_output / d_input)^T.

        The Jacobian J^T is given by (W \odot phi'), see Chen 2018 paper.
        """
        batch = self.activation.grad_phi.size()[0]
        jacobian = einsum(
            'ij,bi->ij',
            (self.linear.weight.t(), self.activation.grad_phi)) / batch
        return jacobian

    def _compute_residuum(self, modify_2nd_order_terms):
        r"""Compute the Hessian residuum accounting for 2nd-order layer effects.

        The residuum (res) is given by:
            res = diag(phi'') \odot grad_output,
        where grad_output denotes the derivative of the loss function with
        respect to the layer output.

        Returns:
        --------
        (torch.Tensor)
            Diagonal term of the quadratic term (corresponding to the only
            nonzero elements)
        """
        residuum_diag = einsum('bi,bi->i',
                               (self.activation.gradgrad_phi,
                                self.activation.grad_output)).detach()
        if modify_2nd_order_terms == 'none':
            pass
        elif modify_2nd_order_terms == 'clip':
            residuum_diag.clamp_(min=0)
        elif modify_2nd_order_terms == 'abs':
            residuum_diag.abs_()
        elif modify_2nd_order_terms == 'zero':
            residuum_diag.zero_()
        else:
            raise ValueError('Unknown 2nd-order term strategy {}'.format(
                modify_2nd_order_terms))
        return residuum_diag

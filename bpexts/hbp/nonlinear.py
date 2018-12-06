"""Base class for elementwise activation layer with HBP functionality."""

from torch import (einsum, diagflat)
from .module import hbp_decorate


def hbp_elementwise_nonlinear(module_subclass):
    """Create new class simplifying the implementation of HBP for a layer
    which applies a nonlinear function phi elementwise to its inputs.
    """
    as_hbp_module = hbp_decorate(module_subclass)

    class HBPElementwiseNonlinear(as_hbp_module):
        """Wrapper around elementwise nonlinearity for HBP.

        It is assumed that the nonlinear layer does not possess any trainable
        parameters.

        For working Hessian backpropagation, the following method should
        be implemented by the user:
        - hbp_derivative_hooks()

        Attributes:
        -----------
        grad_output (torch.Tensor): Gradient with respect to the output
        grad_phi (torch.Tensor): First derivative of function evaluated
                                 on the input, phi'( input )
        gradgrad_phi (torch.Tensor): Second derivative of phi evaluated
                                     on the input, phi''(input)
        """
        __doc__ = as_hbp_module.__doc__

        def hbp_derivative_hooks(self):
            """Register hooks computing first and second derivative of phi.

            The hooks should compute the following buffers:
            1) 'grad_phi': First derivative of nonlinear function applied
                           to the layer input
            2) 'gradgrad_phi': Second derivative of nonlinear function applied
                               to the layer input
            """
            raise NotImplementedError('Please register hooks that track'
                                      ' 1) grad_phi, 2) gradgrad_phi')

        # override
        def hbp_hooks(self):
            """Register hooks required for HBP.

            The following hooks have to be registered:
            1) 'grad_output': Derivative of loss with respect to layer output
            """
            self.register_exts_backward_hook(self.store_grad_output)
            self.hbp_derivative_hooks()

        @staticmethod
        def store_grad_output(module, grad_input, grad_output):
            """Save gradient with respect to output as buffer.

            Intended use as backward hook.
            Initialize module buffer 'grad_output'.
            """
            if not len(grad_output) == 1:
                raise ValueError('Cannot handle multi-output scenario')
            module.register_exts_buffer('grad_output', grad_output[0].detach())

        # override
        def input_hessian(self, output_hessian):
            """Compute input Hessian.

            The input Hessian consists of two parts:
                input_hessian = gauss_newton + residuum.

            For an elementwise nonlinear layer, the following relations hold:
            i) gauss_newton = diag(grad_phi) * output_hessian * diag(grad_phi)
            ii) residuum = diag(gradgrad_phi) \odot grad_output
            """
            in_hessian = self._gauss_newton(output_hessian)
            in_hessian.add_(self._residuum())
            return in_hessian

        def _gauss_newton(self, output_hessian):
            """Compute the Gauss-Newton matrix.

            The Gauss Newton is given by
             jacobian^T * output_hessian * jacobian.
            In this routine, the approximation
             mean(jacobian^T) * output_hessian * mean(jacobian)
            is applied, where mean(.) means batch_averaging.

            Parameters:
            -----------
            output_hessian (torch.Tensor): Hessian with respect to outputs

            Returns:
            --------
            (torch.Tensor): Gauss-Newton matrix, batch-averaged
            """
            # FAR better approximation
            batch = self.grad_phi.size()[0]
            jacobian = self.grad_phi.view(batch, -1)
            return einsum('bi,ij,bj->ij', (jacobian,
                                           output_hessian,
                                           jacobian)) / batch
            # BAD approximation
            # batch_size = self.grad_phi.size()[0]
            # jacobian = self.grad_phi.mean(0)
            # return torch.einsum('i,ij,j->ij',
            #                    (jacobian,
            #                     output_hessian,
            #                     jacobian))

        def _residuum(self):
            """Residuum of the input Hessian matrix.

            The computed relation is
             diag(gradgrad_sigmoid) \odot grad_output.

            If the mode corresponds to BDA-PCH, concavity is eliminated by
            casting the diagonal residuum to its absolute values.

            Returns:
            --------
            (torch.Tensor): Residuum Hessian matrix, averaged over batch
            """
            batch = self.gradgrad_phi.size()[0]
            residuum_diag = einsum('bi,bi->i',
                                   (self.gradgrad_phi.view(batch, -1),
                                    self.grad_output.view(batch, -1)))
            return diagflat(residuum_diag)

    HBPElementwiseNonlinear.__name__ = 'HBPElementwiseNonlinear{}'.format(
            module_subclass.__name__)

    return HBPElementwiseNonlinear

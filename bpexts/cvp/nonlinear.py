"""Base class for elementwise activation layer with CVP functionality."""

from torch import diagflat
from ..hbp.module import hbp_decorate
from ..utils import einsum


def cvp_elementwise_nonlinear(module_subclass):
    """Create new class simplifying the implementation of CVP for a layer
    which applies a nonlinear function phi elementwise to its inputs.
    """
    as_cvp_module = hbp_decorate(module_subclass)

    class CVPElementwiseNonlinear(as_cvp_module):
        """Wrapper around elementwise nonlinearity for CVP.

        It is assumed that the nonlinear layer does not possess any trainable
        parameters.

        For working CVP, the following method should be implemented by the user:
        - cvp_derivative_hooks()

        Attributes:
        -----------
        grad_output (torch.Tensor): Gradient with respect to the output
        grad_phi (torch.Tensor): First derivative of function evaluated
                                 on the input, phi'( input )
        gradgrad_phi (torch.Tensor): Second derivative of phi evaluated
                                     on the input, phi''(input)
        """
        __doc__ = as_cvp_module.__doc__

        def cvp_derivative_hooks(self):
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
            self.cvp_derivative_hooks()

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
        def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
            """Return CVP with respect to the input."""

            def _input_hessian_vp(v):
                """Multiplication by the Hessian w.r.t. the input."""
                return self._input_jacobian_transpose(
                    output_hessian(self._input_jacobian(v))
                ) + self._input_hessian_residuum(v, modify_2nd_order_terms)

            return _input_hessian_vp

        def _input_jacobian(self, v):
            """Apply the Jacobian with respect to the input."""
            batch = self.grad_phi.size(0)
            features = self.grad_phi.numel() // batch
            assert tuple(v.size()) == (batch * features, )
            result = einsum(
                'bj,bj->bj',
                (self.grad_phi.view(batch, -1), v.view(batch, features)))
            assert tuple(result.size()) == (batch, features)
            return result.view(-1)

        def _input_jacobian_transpose(self, v):
            """Apply the transposed Jacobian with respect to the input."""
            batch = self.grad_phi.size(0)
            features = self.grad_phi.numel() // batch
            assert tuple(v.size()) == (batch * features, )
            result = einsum(
                'bj,bj->bj',
                (self.grad_phi.view(batch, -1), v.view(batch, features)))
            assert tuple(result.size()) == (batch, features)
            return result.view(-1)

        def _input_hessian_residuum(self, v, modify_2nd_order_terms):
            """Apply the (modified) residuum of the input Hessian."""
            batch = self.grad_output.size(0)
            features = self.grad_output.numel() // batch
            assert tuple(v.size()) == (batch * features, )
            res_diag = self.gradgrad_phi.view(-1) * self.grad_output.view(-1)
            # different scenarios
            if modify_2nd_order_terms == 'none':
                pass
            elif modify_2nd_order_terms == 'clip':
                res_diag.clamp_(min=0)
            elif modify_2nd_order_terms == 'abs':
                res_diag.abs_()
            elif modify_2nd_order_terms == 'zero':
                res_diag.zero_()
            else:
                raise ValueError('Unknown 2nd-order term strategy {}'.format(
                    modify_2nd_order_terms))
            result = v * res_diag
            assert tuple(result.size()) == (batch * features, )
            return result

    CVPElementwiseNonlinear.__name__ = 'CVPElementwiseNonlinear{}'.format(
        module_subclass.__name__)

    return CVPElementwiseNonlinear

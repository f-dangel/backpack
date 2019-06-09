"""Extend module for HBP."""

import warnings
from ..decorator import decorate


def hbp_decorate(module_subclass):
    """Create new class from torch.nn.Module subclass implementing HBP."""
    decorated_subclass = decorate(module_subclass)

    class HBPModule(decorated_subclass):
        """Wrapper around decorated torch.nn.Module subclass for HBP.

        For working Hessian backpropagation, the following methods should
        be implemented by the user:
        - hbp_hooks()
        - parameter_hessian()
        - input_hessian()
        """

        __doc__ = '[Decorated by bpexts for HBP] {}'.format(
            module_subclass.__doc__)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # determine approximation mode for backward pass in HBP
            self.set_hbp_approximation()
            self.disable_hbp()
            self.enable_hbp()

        def set_hbp_approximation(self,
                                  average_input_jacobian=None,
                                  average_parameter_jacobian=None):
            """Set approximation mode for HBP.

            If ``average_*_jacobians`` is set to ``True``, the Jacobian that
            was averaged in advance will be used to backpropagate the input
            and parameter Hessian.

            Parameters:
            -----------
            average_input_jacobian : bool
                Use batch averaged Jacobian to compute the input Hessian
            average_parameter_jacobian : bool
                Use batch averaged Jacobian to compute the parameter Hessian.
                (Will be ignored if the module does not possess parameters)
            """
            allowed = [True, False, None]
            assert average_input_jacobian in allowed
            assert average_parameter_jacobian in allowed
            self.average_input_jac = average_input_jacobian
            self.average_param_jac = average_parameter_jacobian

        def enable_hbp(self):
            """Enable Hessian backpropagation functionality.

            Set up hooks again.
            """
            self.hbp_hooks()

        def disable_hbp(self, keep_buffers=False):
            """Disable Hessian backpropagation functionality."""
            self.disable_exts(keep_buffers=keep_buffers)

        def hbp_hooks(self):
            """Register all hooks required for Hessian backpropagation.

            This method should be implemented for custom HBP modules.
            """
            warnings.warn('WARNING: Override hbp_hooks if your module'
                          ' requires access to intermediate quantities'
                          ' during Hessian backpropagation.')
            pass

        def backward_hessian(self,
                             output_hessian,
                             compute_input_hessian=True,
                             modify_2nd_order_terms='none'):
            """Propagate Hessian, optionally compute parameter Hessian.

            Backpropagation of the Hessian requires a layer to provide a method
            that, given the Hessian with respect to its output (output_hessian)
            computes the Hessian with respect to its input. During this
            backward procedure, parameter Hessians are computed.

            Classes inheriting from HPBModule must implement this backward
            method.

            Parameters:
            -----------
            output_hessian (torch.Tensor): Hessian with respect to the layer's
                                           output
            compute_input_hessian (bool): Compute the Hessian with respect to
                                          the layer's input (e.g not necessary
                                          for the first layer in a network)
            modify_2nd_order_terms : string ('none', 'clip', 'sign', 'zero')
                String specifying the strategy for dealing with 2nd-order
                module effects (only required if nonlinear layers are involved)

            Returns:
            --------
            input_hessian (torch.Tensor): Hessian with respect to layer input.
                                          If compute_input_hessian is False,
                                          return None.
            """
            self.compute_backward_hessian_quantities()
            if self.has_trainable_parameters():
                self.parameter_hessian(output_hessian)
            return (None
                    if compute_input_hessian is False else self.input_hessian(
                        output_hessian,
                        modify_2nd_order_terms=modify_2nd_order_terms))

        def compute_backward_hessian_quantities(self):
            """Compute quantities required for backprop of the Hessian."""
            pass

        def has_trainable_parameters(self):
            """Check if there are trainable parameters.

            Returns:
            --------
            (bool) there are trainable parameters
            """
            for p in self.parameters():
                if p.requires_grad:
                    return True
            return False

        def parameter_hessian(self, output_hessian):
            """Initialize Hessians with respect to trainable layer parameters.

            Parameters:
            -----------
            output_hessian (torch.Tensor): Hessian matrix of the loss with
                                           respect to the module's output

            If p is a parameter of the model, after calling this method the
            attributes p.hessian should contain a torch.Tensor representation
            of the Hessian of p or a function that, when evaluated, returns
            a tensor representation of this Hessian.

            Moreover, p.hvp should hold a function that provides implicit
            Hessian vector products.
            """
            raise NotImplementedError('Hessian backpropagation modules with'
                                      ' trainable parameters must implement'
                                      ' this method computing Hessians with'
                                      ' respect to their parameters.')

        def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
            """Compute Hessians with respect to layer input.

            Parameters:
            -----------
            output_hessian (torch.Tensor): Hessian matrix of the loss with
                                           respect to the module's output

            Returns:
            --------
            (torch.Tensor): Tensor representation of the loss function's
                            Hessian with respect to the module's input.
            """
            raise NotImplementedError('Hessian backpropagation modules'
                                      'must implement this method to be'
                                      'able to pass Hessians backward')

        def extra_repr(self):
            """Show HBP approximation mode."""
            repr = super().extra_repr()
            if self.average_input_jac is not None:
                repr = '{}, avg_input_jac: {}'.format(repr,
                                                      self.average_input_jac)
            if self.average_param_jac is not None:
                repr = '{}, avg_param_jac: {}'.format(repr,
                                                      self.average_param_jac)
            return repr

        def uses_hbp_approximation(self, average_input_jacobian,
                                   average_parameter_jacobian):
            """Check if module applies the specified HBP approximation."""
            same_param_approx = (
                average_parameter_jacobian == self.average_param_jac)
            same_input_approx = (
                average_input_jacobian == self.average_input_jac)
            return (same_param_approx == same_input_approx)

        # --- hooks ---
        @staticmethod
        def store_input(module, input):
            """Save reference to input of layer.

            Intended use as pre-forward hook.

            Initialize module buffer 'input'.
            """
            if not len(input) == 1:
                raise ValueError('Cannot handle multi-input scenario')
            module.register_exts_buffer('input', input[0])

    HBPModule.__name__ = 'HBP{}'.format(module_subclass.__name__)

    return HBPModule

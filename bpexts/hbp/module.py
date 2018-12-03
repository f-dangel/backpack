#!/usr/bin/python3

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
            self.enable_hbp()

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

        def backward_hessian(self, output_hessian, compute_input_hessian=True):
            """Propagate Hessian, optionally compute parameter Hessians.

            Backpropagation of the Hessian requires a layer to provide a method
            that, given the Hessian with respect to its output (output_hessian)
            computes the Hessian with respect to its input. During this
            backward procedure, parameter Hessians can be computed.

            Classes inheriting from HPBModule must implement this backward
            method.

            Parameters:
            -----------
            output_hessian (torch.Tensor): Hessian with respect to the layer's
                                           output
            compute_input_hessian (bool): Compute the input with respect to the
                                          layer's input (e.g not necessary for
                                          the first layer in a network)

            Returns:
            --------
            input_hessian (torch.Tensor): Hessian with respect to layer input.
                                          If compute_input_hessian is False,
                                          return None.
            """
            if self.has_trainable_parameters():
                self.init_parameter_hessian(output_hessian)
            return None if compute_input_hessian is False else\
                self.input_hessian(output_hessian)

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

        def init_parameter_hessian(output_hessian):
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

        def input_hessian(self, output_hessian):
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

    HBPModule.__name__ = 'HBP{}'.format(module_subclass.__name__)
    return HBPModule

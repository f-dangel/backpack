"""Hessian backpropagation implementation of torch.nn.Sequential."""


from .module import hbp_decorate
from torch.nn import Sequential


class HBPSequential(hbp_decorate(Sequential)):
    """A sequence of HBP modules."""

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def backward_hessian(self, output_hessian, compute_input_hessian=False,
                         modify_2nd_order_terms='none'):
        """Propagate Hessian through the network.

        Starting from the last layer, call `backward_hessian` recursively
        until ending up in the first module.
        """
        out_h = output_hessian
        for idx in reversed(range(len(self))):
                module = self[idx]
                compute_in = True if (idx != 0) else compute_input_hessian
                out_h = module.backward_hessian(
                        out_h,
                        compute_input_hessian=compute_in,
                        modify_2nd_order_terms=modify_2nd_order_terms)
        return out_h

"""
Abstact classfor Hessian backpropagation implementation modules connected
in parallel.

Both versions behave identically in the forward and backward
pass, while the backward pass of the Hessian yields the Hessian
for different groupings of parameters.
"""

from torch.nn import Module
from ..module import hbp_decorate

class HBPParallel(hbp_decorate(Module)):
    """Abstract class for implementing HBP with parameter splitting."""

    def _register_wrapped_module(self, layer):
        """Register the wrapped module."""
        self.add_module('main', layer)

    def _get_parallel_module(self, idx):
        """Return parallel module of index `idx`."""
        return getattr(self.main, self._parallel_module_name(idx))

    @staticmethod
    def _parallel_module_name(idx):
        """Return internal name for parallel layer."""
        return 'parallel{}'.format(idx)

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Use united layer to backward the Hessian."""
        return self.main.input_hessian(
                output_hessian,
                modify_2nd_order_terms=modify_2nd_order_terms)

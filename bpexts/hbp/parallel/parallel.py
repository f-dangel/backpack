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
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks

    def parallel_children(self):
        """Iterate over all parallel children."""
        for i in range(self.num_blocks):
            yield self._get_parallel_module(i)

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
    def backward_hessian(self, output_hessian,
                         compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        self._before_backward_hessian()
        return super().backward_hessian(
                output_hessian,
                compute_input_hessian=compute_input_hessian,
                modify_2nd_order_terms=modify_2nd_order_terms)

    def _before_backward_hessian(self):
        """Do something before HBP."""
        pass

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Use united layer to backward the Hessian."""
        return self.main.input_hessian(
                output_hessian,
                modify_2nd_order_terms=modify_2nd_order_terms)

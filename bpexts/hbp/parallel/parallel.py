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

    def __init__(self, layer, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.create_main_and_children(layer)
        assert hasattr(self, 'main')
        for idx in range(num_blocks):
            assert hasattr(self.main, self._parallel_module_name(idx))
        self.set_hbp_approximation()

    def forward(self, input):
        """Feed through main layer."""
        return self.main(input)

    def create_main_and_children(self, layer):
        raise NotImplementedError

    # override
    def enable_hbp(self):
        """Enable HBP hooks of main, disable for children."""
        try:
            self.main.enable_hbp()
        except AttributeError:
            pass
        try:
            for mod in self.parallel_children():
                mod.disable_hbp()
        except AttributeError:
            pass

    # override
    def uses_hbp_approximation(self, average_input_jacobian,
                               average_parameter_jacobian):
        """Check if module applies the specified HBP approximation."""
        uses_same = [
            self.main.uses_hbp_approximation(average_input_jacobian,
                                             average_parameter_jacobian)
        ]
        for mod in self.parallel_children():
            uses_same.append(
                mod.uses_hbp_approximation(average_input_jacobian,
                                           average_parameter_jacobian))
        return all(uses_same)

    # override
    def set_hbp_approximation(self,
                              average_input_jacobian=None,
                              average_parameter_jacobian=True):
        try:
            self.main.set_hbp_approximation(
                average_input_jacobian=average_input_jacobian,
                average_parameter_jacobian=average_parameter_jacobian)
        except AttributeError:
            pass
        try:
            for mod in self.parallel_children():
                mod.set_hbp_approximation(
                    average_input_jacobian=average_input_jacobian,
                    average_parameter_jacobian=average_parameter_jacobian)
        except AttributeError:
            pass
        super().set_hbp_approximation(
            average_input_jacobian=None, average_parameter_jacobian=None)

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
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Use united layer to backward the Hessian."""
        return self.main.input_hessian(
            output_hessian, modify_2nd_order_terms=modify_2nd_order_terms)

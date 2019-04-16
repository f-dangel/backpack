"""Curvature-vector products for sequences of modules."""

from torch.nn import Sequential
from ..hbp.module import hbp_decorate


class CVPSequential(hbp_decorate(Sequential)):
    """Sequence of modules with recursive Hessian-vector products."""

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def backward_hessian(self,
                         output_hessian,
                         compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        """Propagate Hessian-vector product through the network.

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

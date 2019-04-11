"""Curvature-vector products for linear layer."""

from torch import einsum
from torch.nn import CrossEntropyLoss, functional
from ..hbp.module import hbp_decorate


class CVPCrossEntropyLoss(hbp_decorate(CrossEntropyLoss)):
    """Cross-entropy loss with recursive Hessian-vector products."""

    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        if weight is not None:
            raise NotImplementedError('Only supports weight = None')
        if ignore_index != -100:
            raise NotImplementedError('Only supports ignore_index = -100')
        if reduce is not None:
            raise NotImplementedError('Only supports reduce = None')
        if reduction is not 'mean':
            raise NotImplementedError(r"Only supports reduction = 'mean'")
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    # override
    def hbp_hooks(self):
        """Install hooks to track quantities required for CVP."""
        self.register_exts_forward_pre_hook(self.compute_and_store_softmax)

    # --- hooks ---
    @staticmethod
    def compute_and_store_softmax(module, input):
        """Compute and save softmax of layer input.

        Intended use as pre-forward hook.
        Initialize module buffer 'input_softmax'.
        """
        if not len(input) == 2:
            raise ValueError('Wrong number of inputs')
        assert len(tuple(input[0].size())) == 2
        input_softmax = functional.softmax(input[0].detach(), dim=1)
        assert input_softmax.size() == input[0].size()
        module.register_exts_buffer('input_softmax', input_softmax)

    # --- end of hooks ---

    # --- Hessian-vector product with the input Hessian ---
    # override
    def input_hessian(self, output_hessian=None,
                      modify_2nd_order_terms='none'):
        """Return CVP with respect to the input."""
        if output_hessian is not None:
            raise ValueError('No output Hessian required for loss functions')
        batch, num_classes = tuple(self.input_softmax.size())

        def _input_hessian_vp(v):
            """Multiplication by the Hessian w.r.t. the input."""
            assert tuple(v.size()) == (self.input_softmax.numel(), )
            result = v.view(batch, num_classes) / batch
            result = einsum(
                'bi,bi->bi', (self.input_softmax, result)) - einsum(
                    'bi,bj,bj->bi',
                    (self.input_softmax, self.input_softmax, result))
            assert tuple(result.size()) == (batch, num_classes)
            return result.view(-1)

        return _input_hessian_vp

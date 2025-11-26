"""Define layer extensions with derivatives based on autodiff."""

from backpack.core.derivatives.automatic import AutomaticDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNExactAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for modules via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(AutomaticDerivatives(), sum_batch=True)


class DiagGGNExactLinearAutomatic(DiagGGNBaseModule):
    """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(
            AutomaticDerivatives(), params=["weight", "bias"], sum_batch=True
        )


class BatchDiagGGNExactAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for modules via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(AutomaticDerivatives(), sum_batch=False)


class BatchDiagGGNExactLinearAutomatic(DiagGGNBaseModule):
    """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(
            AutomaticDerivatives(), params=["weight", "bias"], sum_batch=False
        )


class BatchGradLinearAutomatic(BatchGradBase):
    """Batch gradients for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(AutomaticDerivatives(), params=["weight", "bias"])

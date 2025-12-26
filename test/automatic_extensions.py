"""Define layer extensions with derivatives based on autodiff."""

from test.automatic_derivatives import (
    LinearAutomaticDerivatives,
    ReLUAutomaticDerivatives,
    SigmoidAutomaticDerivatives,
)

from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNExactReLUAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for ``torch.nn.ReLU`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(ReLUAutomaticDerivatives(), sum_batch=True)


class DiagGGNExactSigmoidAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for ``torch.nn.Sigmoid`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(SigmoidAutomaticDerivatives(), sum_batch=True)


class DiagGGNExactLinearAutomatic(DiagGGNBaseModule):
    """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(
            LinearAutomaticDerivatives(), params=["weight", "bias"], sum_batch=True
        )


class BatchDiagGGNExactReLUAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for ``torch.nn.ReLU`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(ReLUAutomaticDerivatives(), sum_batch=False)


class BatchDiagGGNExactSigmoidAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for ``torch.nn.Sigmoid`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(SigmoidAutomaticDerivatives(), sum_batch=False)


class BatchDiagGGNExactLinearAutomatic(DiagGGNBaseModule):
    """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(
            LinearAutomaticDerivatives(), params=["weight", "bias"], sum_batch=False
        )


class BatchGradLinearAutomatic(BatchGradBase):
    """Batch gradients for ``torch.nn.Linear`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(LinearAutomaticDerivatives(), params=["weight", "bias"])

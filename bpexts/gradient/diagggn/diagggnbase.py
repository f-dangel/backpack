from ..extensions import DIAG_GGN
from ..matbackprop import BackpropSqrtMatrixWithJacobian

BACKPROPAGATED_MATRIX_SQRT_NAME = "_backpropagated_sqrt_ggn"
EXTENSION = DIAG_GGN


class DiagGGNBase(BackpropSqrtMatrixWithJacobian):
    def __init__(self, params=[]):
        super().__init__(
            BACKPROPAGATED_MATRIX_SQRT_NAME, EXTENSION, params=params)

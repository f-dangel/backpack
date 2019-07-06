from ...context import CTX
from ...extensions import KFLR
from ...matbackprop import BackpropSqrtMatrixWithJacobian

BACKPROPAGATED_MATRIX_SQRT_NAME = "_kflr_backpropagated_sqrt_ggn"
EXTENSION = DIAG_GGN


class KFLRBase(BackpropSqrtMatrixWithJacobian):
    def __init__(self, params=[]):
        super().__init__(
            BACKPROPAGATED_MATRIX_SQRT_NAME, EXTENSION, params=params)

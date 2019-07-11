from ...extensions import KFLR
from ...approx import MatToJacMatJac

BACKPROPAGATED_MATRIX_SQRT_NAME = "_kflr_backpropagated_sqrt_ggn"
EXTENSION = KFLR


class KFLRBase(MatToJacMat):
    def __init__(self, params=[]):
        super().__init__(
            BACKPROPAGATED_MATRIX_SQRT_NAME, EXTENSION, params=params)

from ...extensions import KFAC
from ...matbackprop import MatToJacMat

BACKPROPAGATED_MATRIX_SQRT_NAME = "_kfac_backpropagated_sqrt_ggn"
EXTENSION = KFAC


class KFACBase(MatToJacMat):
    def __init__(self, params=[]):
        super().__init__(
            BACKPROPAGATED_MATRIX_SQRT_NAME, EXTENSION, params=params)

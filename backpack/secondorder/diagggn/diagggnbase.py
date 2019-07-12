from ...matbackprop import MatToJacMat
from ...extensions import DIAG_GGN


class DiagGGNBase(MatToJacMat):
    MAT_NAME_IN_CTX = "_backpropagated_sqrt_ggn"
    EXTENSION = DIAG_GGN

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

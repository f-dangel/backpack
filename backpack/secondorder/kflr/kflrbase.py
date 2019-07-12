from ...extensions import KFLR
from ...matbackprop import MatToJacMat


class KFLRBase(MatToJacMat):
    MAT_NAME_IN_CTX = "_kflr_backpropagated_sqrt_ggn"
    EXTENSION = KFLR

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

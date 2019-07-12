from ...extensions import KFAC
from ...matbackprop import MatToJacMat


class KFACBase(MatToJacMat):
    MAT_NAME_IN_CTX = "_kfac_backpropagated_sqrt_ggn"
    EXTENSION = KFAC

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

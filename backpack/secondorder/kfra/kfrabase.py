from ...extensions import KFRA
from ...approx import MatToJacMatJac


class KFRABase(MatToJacMat):
    MAT_NAME_IN_CTX = "_kfra_backpropagated_sqrt_ggn"
    EXTENSION = KFRA

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

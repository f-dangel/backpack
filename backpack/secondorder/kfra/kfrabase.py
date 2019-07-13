from ...extensions import KFRA
from ...matbackprop import ExpectationApproximationMatToJacMatJac


class KFRABase(ExpectationApproximationMatToJacMatJac):
    MAT_NAME_IN_CTX = "_kfra_backpropagated_ea_h"
    EXTENSION = KFRA

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

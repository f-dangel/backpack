from backpack.extensions.mat_to_mat_jac_base import MatToJacMat


class DiagGGNBaseModule(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives, params=params)

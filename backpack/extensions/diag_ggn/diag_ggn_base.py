from backpack.extensions.newmatbackprop import MatToJacMat


class DiagGGNBaseModule(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives, params=params)

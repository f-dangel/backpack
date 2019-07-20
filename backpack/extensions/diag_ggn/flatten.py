from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNFlatten(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=None)

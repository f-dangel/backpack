from backpack.core.derivatives.identity import IdentityDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNIdentity(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=IdentityDerivatives())

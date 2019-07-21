from .diag_h_base import DiagHBaseModule


class DiagHFlatten(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=None)

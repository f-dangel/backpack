from .hbpbase import HBPBaseModule


class HBPFlatten(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=None)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        return backproped

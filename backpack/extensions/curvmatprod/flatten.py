from .cmpbase import CMPBase


class CMPFlatten(CMPBase):
    def __init__(self):
        super().__init__(derivatives=None)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        return backproped

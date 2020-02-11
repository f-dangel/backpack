from backpack.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):
        return None

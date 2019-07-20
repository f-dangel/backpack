from backpack.new_extensions.module_extension import ModuleExtension


class FirstOrderExtension(ModuleExtension):

    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):
        return None

class CTX:
    """
    Global Class holding the configuration of the backward pass
    """

    active_exts = tuple()
    debug = False

    @staticmethod
    def set_active_exts(active_exts):
        CTX.active_exts = tuple()
        for act_ext in active_exts:
            CTX.active_exts += (act_ext,)

    @staticmethod
    def get_active_exts():
        return CTX.active_exts

    @staticmethod
    def add_hook_handle(hook_handle):
        if getattr(CTX, "hook_handles", None) is None:
            CTX.hook_handles = []
        CTX.hook_handles.append(hook_handle)

    @staticmethod
    def remove_hooks():
        for handle in CTX.hook_handles:
            handle.remove()
        CTX.hook_handles = []

    @staticmethod
    def is_extension_active(*extension_classes):
        for backpack_ext in CTX.get_active_exts():
            if isinstance(backpack_ext, extension_classes):
                return True
        return False

    @staticmethod
    def get_debug():
        return CTX.debug

    @staticmethod
    def set_debug(debug):
        CTX.debug = debug

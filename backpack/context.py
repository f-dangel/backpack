from .extensions import Extension

RETURN_IF_NOT_EXISTING = None


def get_from_ctx(name):
    value = CTX.backpropQuantities.get(name, RETURN_IF_NOT_EXISTING)
    if value is RETURN_IF_NOT_EXISTING:
        raise ValueError("The attribute {} does not exist in CTX".format(name))
    return value


def set_in_ctx(name, value):
    CTX.backpropQuantities[name] = value


class CTX:
    """
    Global Class holding the configuration of the backward pass
    """
    activeExts = tuple()
    backpropQuantities = {}

    @staticmethod
    def set_active_exts(active_exts):
        CTX.activeExts = tuple()
        for act_ext in active_exts:
            if not isinstance(act_ext, Extension):
                raise ValueError("Unknown extension ... was expecting ...")
            CTX.activeExts += (act_ext, )

    @staticmethod
    def active_exts():
        return CTX.activeExts

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
    def clear():
        del CTX.backpropQuantities
        CTX.backpropQuantities = {}

import warnings


def get_from_ctx(name):
    value = CTX.backproped_quantities.get(name, None)
    if value is None:
        warnings.warn("The attribute {} does not exist in CTX".format(name))
    return value


def set_in_ctx(name, value):
    CTX.backproped_quantities[name] = value


class CTX:
    """
    Global Class holding the configuration of the backward pass
    """
    active_exts = tuple()
    backproped_quantities = {}

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
    def clear():
        del CTX.backproped_quantities
        CTX.backproped_quantities = {}

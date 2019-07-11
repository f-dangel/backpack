from .extensions import Extensions


class CTX:
    """
    Global Class holding the configuration of the backward pass
    """
    activeExtsDict = {}

    @staticmethod
    def as_dict():
        return {ext: CTX.activeExtsDict.get(ext, False) for ext in Extensions.ext_list()}

    @staticmethod
    def from_dict(dic):
        for key, val in dic.items():
            CTX.activeExtsDict[key] = val

    @staticmethod
    def is_active(ext):
        Extensions.check_exists(ext)
        return CTX.activeExtsDict.get(ext, False)

    @staticmethod
    def active_exts():
        return [ext for ext, active in CTX.as_dict().items() if active]

    @staticmethod
    def add_hook_handle(hook_handle):
        if getattr(CTX.activeExtsDict, "hook_handles", None) is None:
            CTX.hook_handles = []
        CTX.hook_handles.append(hook_handle)

    @staticmethod
    def remove_hooks():
        for handle in CTX.hook_handles:
            handle.remove()
        CTX.hook_handles = []

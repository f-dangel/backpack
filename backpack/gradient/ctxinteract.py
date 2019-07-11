from .context import CTX


class ActOnCTX():
    def __init__(self, ctx_name):
        self.__CTX_NAME = ctx_name

    def get_from_ctx(self):
        value = getattr(CTX, self.__CTX_NAME, None)
        if value is None:
            raise ValueError(
                "Attribute {} for backpropagation does not exist in CTX".
                format(self.__CTX_NAME))
        return value

    def set_in_ctx(self, value):
        setattr(CTX, self.__CTX_NAME, value)

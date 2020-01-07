class HiddenLayer:
    def __init__(self, linear_cls, activation_cls=None):
        self.linear_cls = linear_cls
        self.activation_cls = activation_cls

    def _has_activation(self):
        return self.activation_cls is not None

    def get_modules(self, settings):
        modules = [self._linear_layer(settings)]
        if self._has_activation():
            modules.append(self._activation_layer())
        return modules

    def _linear_layer(self, settings):
        return self.linear_cls(
            in_features=settings["in_features"],
            out_features=settings["out_features"],
            bias=settings["bias"],
        )

    def _activation_layer(self):
        return self.activation_cls()


class HiddenLayer2(HiddenLayer):
    def _linear_layer(self, settings):
        return self.linear_cls(
            in_features=settings["out_features"],
            out_features=settings["out_features2"],
            bias=settings["bias"],
        )


def single_linear_layer(settings, linear_cls, activation_cls=None):
    """Linear(x,y) -> (optional: Activation)"""
    hidden = HiddenLayer(linear_cls, activation_cls=activation_cls)
    return hidden.get_modules(settings)


def two_linear_layers(settings, linear_cls, activation_cls=None):
    """Linear(x,y) -> (optional: Activation) -> Linear(y,z)"""
    hidden1 = HiddenLayer(linear_cls, activation_cls=activation_cls)
    hidden2 = HiddenLayer2(linear_cls)

    return hidden1.get_modules(settings) + hidden2.get_modules(settings)

import torch
from .test_problem import TestProblem
from backpack import extend


class ProblemBase():
    def __init__(self, settings):
        self.settings = settings

    def __call__(self):
        model = self.get_extended_model()
        X, Y = self.get_XY(model)
        loss_func = extend(self.get_loss_func())
        return TestProblem(X, Y, model, loss_func)

    def get_extended_model(self):
        modules = self.get_modules()
        return self.extend_and_create_sequential(modules)

    def get_modules(self):
        raise NotImplementedError

    def get_XY(self, model):
        raise NotImplementedError

    def get_loss_func(self):
        raise NotImplementedError

    @staticmethod
    def extend_and_create_sequential(modules):
        extended_modules = [extend(m) for m in modules]
        return torch.nn.Sequential(*extended_modules)


class Regression(ProblemBase):
    def get_loss_func(self):
        return torch.nn.MSELoss()

    def get_XY(self, model):
        input_size = (self.settings["batch"], self.settings["in_features"])
        X = torch.randn(size=input_size)
        Y = torch.randn(size=(model(X).shape[0], 1))
        return X, Y


class Classification(ProblemBase):
    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss()

    def get_XY(self, model):
        input_size = (self.settings["batch"], self.settings["in_features"])
        X = torch.randn(size=input_size)
        Y = torch.randint(high=model(X).shape[1], size=(X.shape[0], ))
        return X, Y


class HiddenLayer():
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


class SumOutputLayer():
    def __init__(self, linear_cls):
        self.linear_cls = linear_cls

    def get_modules(self, settings):
        module = self.linear_cls(
            in_features=settings["out_features"],
            out_features=1,
            bias=True,
        )
        return [module]


# To create the test problems


class RegressionSingleLayer(Regression):
    """Linear(x,y) -> (optional: Activation) -> Linear(y, 1) -> MSE """

    def __init__(self, settings, linear_cls, activation_cls=None):
        super().__init__(settings)
        self.hidden = HiddenLayer(linear_cls, activation_cls=activation_cls)
        self.sum_output = SumOutputLayer(linear_cls)

    def get_modules(self):
        return (self.hidden.get_modules(self.settings) +
                self.sum_output.get_modules(self.settings))


class ClassificationSingleLayer(Classification):
    """Linear(x,y) -> (optional: Activation) -> CrossEntropyLoss"""

    def __init__(self, settings, linear_cls, activation_cls=None):
        super().__init__(settings)
        self.hidden = HiddenLayer(linear_cls, activation_cls=activation_cls)

    def get_modules(self):
        return self.hidden.get_modules(self.settings)


class ClassificationTwoLayers(Classification):
    """Linear(x,y) -> (optional: Activation) -> Linear(y,z) -> CrossEntropyLoss"""

    def __init__(self, settings, linear_cls, activation_cls=None):
        super().__init__(settings)
        self.hidden1 = HiddenLayer(linear_cls, activation_cls=activation_cls)
        self.hidden2 = HiddenLayer2(linear_cls)

    def get_modules(self):
        return (self.hidden1.get_modules(self.settings) +
                self.hidden2.get_modules(self.settings))

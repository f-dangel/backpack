import torch

from backpack import extend

from .test_problem import TestProblem


class ProblemBase:
    def __init__(self, input_shape, network_modules):
        self.input_shape = input_shape
        self.net_modules = network_modules

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

    def get_network_modules(self):
        return self.net_modules

    def forward_pass_through_net_only(self):
        net = torch.nn.Sequential(*self.get_network_modules())
        X, _ = self.get_XY(net)
        return net(X)

    def get_num_network_outputs(self):
        output = self.forward_pass_through_net_only()
        return output.numel() // output.shape[0]

    def get_XY(self, model):
        raise NotImplementedError

    def get_loss_func(self):
        raise NotImplementedError

    @staticmethod
    def extend_and_create_sequential(modules):
        sequential = torch.nn.Sequential(*modules)
        return extend(sequential)


class Regression(ProblemBase):
    """(network) -> Flatten -> Linear(x, 1) -> MSE"""

    def get_loss_func(self):
        return torch.nn.MSELoss()

    def get_modules(self):
        modules = self.get_network_modules()
        modules.append(torch.nn.Flatten())
        modules.append(self.sum_output_layer())
        return modules

    def sum_output_layer(self):
        num_outputs = self.get_num_network_outputs()
        return torch.nn.Linear(in_features=num_outputs, out_features=1, bias=True)

    def get_XY(self, model):
        X = torch.randn(size=self.input_shape)
        Y = torch.randn(size=(model(X).shape[0], 1))
        return X, Y


class Classification(ProblemBase):
    """(network) -> Flatten -> CrossEntropy"""

    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss()

    def get_modules(self):
        modules = self.get_network_modules()
        modules.append(torch.nn.Flatten())
        return modules

    def get_XY(self, model):
        X = torch.randn(size=self.input_shape)
        Y = torch.randint(high=model(X).shape[1], size=(X.shape[0],))
        return X, Y


def make_regression_problem(input_shape, modules):
    regression = Regression(input_shape, modules)
    return regression()


def make_classification_problem(input_shape, modules):
    classification = Classification(input_shape, modules)
    return classification()

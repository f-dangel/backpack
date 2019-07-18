from .problems import (RegressionSingleLayer, ClassificationSingleLayer,
                       ClassificationTwoLayers)


def make_regression_problem(activation_cls, linear_cls, settings):
    regression = RegressionSingleLayer(
        settings, linear_cls, activation_cls=activation_cls)
    return regression()


def make_classification_problem(activation_cls, linear_cls, settings):
    classification = ClassificationSingleLayer(
        settings, linear_cls, activation_cls=activation_cls)
    return classification()


def make_2layer_classification_problem(activation_cls, linear_cls, settings):
    classification = ClassificationTwoLayers(
        settings, linear_cls, activation_cls=activation_cls)
    return classification()

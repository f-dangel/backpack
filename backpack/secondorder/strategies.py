class LossHessianStrategy():
    EXACT = "exact"
    SAMPLING = "sampling"
    AVERAGE = "average"
    CHOICES = [
        EXACT,
        SAMPLING,
        AVERAGE,
    ]

    @classmethod
    def check_exists(cls, strategy):
        if not strategy in cls.CHOICES:
            raise AttributeError(
                "Unknown loss Hessian strategy: {}. Expecting one of {}".
                format(strategy, cls.CHOICES))


class BackpropStrategy():
    SQRT = "sqrt"
    BATCH_AVERAGE = "average"

    CHOICES = [
        BATCH_AVERAGE,
        SQRT,
    ]

    @classmethod
    def is_batch_average(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.BATCH_AVERAGE

    @classmethod
    def is_sqrt(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.SQRT

    @classmethod
    def check_exists(cls, strategy):
        if not strategy in cls.CHOICES:
            raise AttributeError(
                "Unknown backpropagation strategy: {}. Expect {}".format(
                    strategy, cls.CHOICES))


class ExpectationApproximation():
    BOTEV_MARTENS = "E[J^T E(H) J]"
    CHEN = "E(J^T) E(H) E(J)"

    CHOICES = [
        BOTEV_MARTENS,
        CHEN,
    ]

    CURRENT = BOTEV_MARTENS

    @classmethod
    def get_current(cls):
        return cls.CURRENT

    @classmethod
    def set_strategy(cls, strategy):
        cls.__check_exists(strategy)
        cls.CURRENT = strategy

    @classmethod
    def should_average_param_jac(cls):
        return cls.CURRENT == cls.CHEN

    @classmethod
    def __check_exists(cls, strategy):
        if not strategy in cls.CHOICES:
            raise AttributeError(
                "Unknown loss Hessian strategy: {}. Expecting one of {}".
                format(strategy, cls.CHOICES))

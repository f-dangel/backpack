class LossHessianStrategy():
    EXACT = "exact"
    SAMPLING = "sampling"
    AVERAGE = "average"
    CHOICES = [
        EXACT,
        SAMPLING,
        AVERAGE,
    ]

    CURRENT = EXACT

    @classmethod
    def get_current(cls):
        return cls.CURRENT

    @classmethod
    def set_strategy(cls, strategy):
        cls.__check_exists(strategy)
        cls.CURRENT = strategy

    @classmethod
    def __check_exists(cls, strategy):
        if not strategy in cls.CHOICES:
            raise AttributeError(
                "Unknown loss Hessian strategy: {}. Expecting one of {}".
                format(which, cls.CHOICES))

    @classmethod
    def is_kfac(cls):
        return cls.CURRENT == cls.SAMPLING

    @classmethod
    def is_kflr(cls):
        return cls.CURRENT == cls.EXACT

    @classmethod
    def is_kfra(cls):
        return cls.is_ea()

    @classmethod
    def is_ea(cls):
        return cls.CURRENT == cls.AVERAGE


class BackpropStrategy():
    SQRT = "sqrt"
    BATCH_AVERAGE = "average"

    CHOICES = [
        BATCH_AVERAGE,
        SQRT,
    ]

    CURRENT = SQRT

    @classmethod
    def is_batch_average(cls):
        return cls.CURRENT == cls.BATCH_AVERAGE

    @classmethod
    def is_sqrt(cls):
        return cls.CURRENT == cls.SQRT

    @classmethod
    def get_current(cls):
        return cls.CURRENT

    @classmethod
    def set_strategy(cls, strategy):
        cls.__check_exists(strategy)
        cls.CURRENT = CURRENT

    @classmethod
    def __check_exists(cls, strategy):
        if not strategy in cls.CHOICES:
            raise AttributeError(
                "Unknown loss Hessian strategy: {}. Expecting one of {}".
                format(which, cls.CHOICES))


class ExpectationApproximation():
    BOTEV_MARTENS = "E[J^T E(H) J]"
    CHEN = "E(J^T) E(H) E(J)"

    CHOICES = [
        BOTEV_MARTENS,
        CHEN,
    ]

    CURRENT = BOTEV_MARTENS

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
                format(which, cls.CHOICES))

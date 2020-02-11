class LossHessianStrategy:
    EXACT = "exact"
    SAMPLING = "sampling"
    SUM = "sum"

    CHOICES = [
        EXACT,
        SAMPLING,
        SUM,
    ]

    @classmethod
    def check_exists(cls, strategy):
        if strategy not in cls.CHOICES:
            raise AttributeError(
                "Unknown loss Hessian strategy: {}. ".format(strategy)
                + "Expecting one of {}".format(cls.CHOICES)
            )


class BackpropStrategy:
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
        if strategy not in cls.CHOICES:
            raise AttributeError(
                "Unknown backpropagation strategy: {}. ".format(strategy)
                + "Expect {}".format(cls.CHOICES)
            )


class ExpectationApproximation:
    BOTEV_MARTENS = "E[J^T E(H) J]"
    CHEN = "E(J^T) E(H) E(J)"

    CHOICES = [
        BOTEV_MARTENS,
        CHEN,
    ]

    @classmethod
    def should_average_param_jac(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.CHEN

    @classmethod
    def check_exists(cls, strategy):
        if strategy not in cls.CHOICES:
            raise AttributeError(
                "Unknown EA strategy: {}. ".format(strategy)
                + "Expect {}".format(cls.CHOICES)
            )

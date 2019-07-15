class ExpectationApproximation():
    AVG_PARAM_JAC = False

    @classmethod
    def set_expectation(avg_param_jac):
        cls.AVG_PARAM_JAC = avg_param_jac

    @classmethod
    def should_average_param_jac(cls):
        return cls.AVG_PARAM_JAC

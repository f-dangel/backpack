"""Passes the calls for the parameters to the derivatives class."""
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):
    """Passes the calls for the parameters to the derivatives class.

    This class implements functions with method name from params.
    If child class wants to overwrite these methods,
    this class needs to check if the method is already implemented.
    """

    def __init__(self, derivatives, params=None):
        """Initializes all methods.

        Args:
            derivatives
                (backpack.core.derivatives.basederivatives.BaseParameterDerivatives):
                Derivatives object assigned to self.derivatives.
            params (list[str]): list of strings with parameter names. Defaults to None.
        """
        self.derivatives = derivatives
        for param_str in params:
            self.__setattr__(param_str, self._make_function(param_str))
        super().__init__(params=params)

    def _make_function(self, param):
        def function(ext, module, g_inp, g_out, bpQuantities):
            return getattr(self.derivatives, f"{param}_jac_t_mat_prod")(
                module, g_inp, g_out, g_out[0], sum_batch=False
            )

        return function

"""Passes the calls for the parameters to the derivatives class."""
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):
    """Passes the calls for the parameters to the derivatives class.

    This class implements functions with method names from params.

    If child class wants to overwrite these methods
    - for example to support an additional external module -
    it can do so using the interface for parameter "param1"
    param1(ext, module, g_inp, g_out, bpQuantities):
        return batch_grads
    In this case, the method is not overwritten by this class.
    """

    def __init__(self, derivatives, params=None):
        """Initializes all methods.

        If the param method has already been defined, it is left unchanged.

        Args:
            derivatives(backpack.core.derivatives.basederivatives.BaseParameterDerivatives):
                Derivatives object assigned to self.derivatives.
            params (list[str]): list of strings with parameter names. Defaults to None.
        """
        self.derivatives = derivatives
        for param_str in params:
            if not hasattr(self, param_str):
                self.__setattr__(param_str, self._make_function(param_str))
        super().__init__(params=params)

    def _make_function(self, param):
        def function(ext, module, g_inp, g_out, bpQuantities):
            return getattr(self.derivatives, f"{param}_jac_t_mat_prod")(
                module, g_inp, g_out, g_out[0], sum_batch=False
            )

        return function

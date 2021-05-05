"""Calculates the batch_grad derivative."""
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):
    """Calculates the batch_grad derivative.

    Passes the calls for the parameters to the derivatives class.
    Implements functions with method names from params.

    If child class wants to overwrite these methods
    - for example to support an additional external module -
    it can do so using the interface for parameter "param1"

    param1(ext, module, g_inp, g_out, bpQuantities):
        return batch_grads

    In this case, the method is not overwritten by this class.
    """

    def __init__(self, derivatives, params):
        """Initializes all methods.

        If the param method has already been defined, it is left unchanged.

        Args:
            derivatives(backpack.core.derivatives.basederivatives.BaseParameterDerivatives):# noqa
                Derivatives object assigned to self.derivatives.
            params (list[str]): list of strings with parameter names.
                For each, a method is assigned.
        """
        self.derivatives = derivatives
        for param_str in params:
            if not hasattr(self, param_str):
                setattr(self, param_str, self._make_param_function(param_str))
        super().__init__(params=params)

    def _make_param_function(self, param):
        """Creates a function that calculates batch_grad wrt param.

        Args:
            param(str): name of parameter

        Returns:
            function: function that calculates batch_grad wrt param
        """

        def param_function(ext, module, g_inp, g_out, bpQuantities):
            """Calculates batch_grad with the help of derivatives object.

            Args:
                ext(backpack.extensions.BatchGrad): extension that is used
                module(torch.nn.Module): module that performed forward pass
                g_inp(tuple[torch.Tensor]): input gradient tensors
                g_out(tuple[torch.Tensor]): output gradient tensors
                bpQuantities(None): additional quantities for second order

            Returns:
                torch.Tensor: scaled individual gradients
            """
            return getattr(self.derivatives, f"{param}_jac_t_mat_prod")(
                module, g_inp, g_out, g_out[0], sum_batch=False
            )

        return param_function

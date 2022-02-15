"""Extension hooks to compact BackPACK quantities during backpropagation."""


class ExtensionHookManager:
    """Manages extension hook calls during backprop. Calls every hook once per param."""

    def __init__(self, *hooks):
        """Store parameter hooks.

        Args:
            hooks (list(callable)): List of functions that accept a tensor and perform
                a side effect. The signature is ``torch.Tensor -> None``.
        """
        self.hooks = hooks
        self.params_visited = set()

    def __call__(self, module):
        """Apply every hook to the module parameters Skip visited parameters.

        This function is handed to the ``backpack`` context manager.

        Args:
            module (torch.nn.Module): The neural network layer that all parameter
                hooks will be applied to.
        """
        if self._should_run_hooks_on_module(module):
            for param in module.parameters():
                if self._should_run_hooks_on_param(param):
                    self._run_hooks(param)

    def _should_run_hooks_on_module(self, module):
        """Check if hooks should be executed on a module.

        Hooks are only executed if the module does not have children.
        This is because due to the execution order of BackPACK's extensions,
        BackPACK quantities have not yet been computed for the children modules
        when the containing module's extension hook is called.

        Args:
            module (torch.nn.Module): Layer of a neural net.

        Returns:
            bool: Whether hooks should be run.
        """
        no_children = len(list(module.children())) == 0
        return no_children

    def _should_run_hooks_on_param(self, param):
        """Check if hooks should be executed on a parameter.

        Hooks are only executed once on every trainable parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns:
            bool: Whether hooks should be run.
        """
        return param.requires_grad and id(param) not in self.params_visited

    def _run_hooks(self, param):
        """Execute all hooks on a parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.
        """
        for hook in self.hooks:
            hook(param)
        self.params_visited.add(id(param))


class ParameterHook:
    """Extension hook class to perform actions on parameters."""

    def __init__(self, savefield):
        self.savefield = savefield

    def __call__(self, param):
        value = self.hook(param)
        self.__save(value, param)

    def hook(self, param):
        """Extract info from a parameter during backpropagation with BackPACK."""
        raise NotImplementedError

    def __save(self, value, param):
        setattr(param, self.savefield, value)


class BatchL2GradHook(ParameterHook):
    """Computes individual gradient squared ℓ₂ norms from individual gradients.

    Requires access to a parameter's ``.grad_batch`` field which is set by the
    ``BatchGrad`` extension.
    """

    def __init__(self, savefield="batch_l2_hook"):
        super().__init__(savefield=savefield)

    def hook(self, param):
        """Extract individual gradient squared ℓ₂ norms from individual gradients."""
        feature_dims = [i + 1 for i in range(param.dim())]
        return (param.grad_batch**2).sum(feature_dims)


class SumGradSquaredHook(ParameterHook):
    """Computes individual gradient second moment from individual gradients.

    Requires access to a parameter's ``.grad_batch`` field which is set by the
    ``BatchGrad`` extension.
    """

    def __init__(self, savefield="sum_grad_squared_hook"):
        super().__init__(savefield=savefield)

    def hook(self, param):
        """Extract individual gradient squared ℓ₂ norms from individual gradients."""
        N_axis = 0
        return (param.grad_batch**2).sum(N_axis)

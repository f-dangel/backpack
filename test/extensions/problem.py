"""Convert problem settings."""

from warnings import warn


class ExtensionsTestProblem:
    def __init__(
        self, input_fn, module_fn, loss_function_fn, target_fn, device, seed, id_prefix,
    ):
        """Collection of information required to test extensions.

        Args:
            input_fn (callable): Function returning the network input.
            module_fn (callable): Function returning the network.
            loss_function_fn (callable): Function returning the loss module.
            target_fn (callable): Function returning the labels.
            device (torch.device): Device to run on.
            seed (int): Random seed.
            id_prefix (str): Extra string added to test id.
        """
        self.module_fn = module_fn
        self.input_fn = input_fn
        self.loss_function_fn = loss_function_fn
        self.target_fn = target_fn

        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix

    def set_up(self):
        warn("Dummy")

    def tear_down(self):
        warn("Dummy")

    def make_id(self):
        """Needs to function without call to `set_up`."""
        warn("Dummy")
        return "dummy"

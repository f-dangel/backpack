"""Convert problem settings."""

import copy
from test.core.derivatives.utils import get_available_devices

import torch

from backpack import extend


def make_test_problems(settings):
    """Creates test problems from settings.

    Args:
        settings (list[dict]): raw settings of the problems

    Returns:
        list[ExtensionTestProblem]
    """
    problem_dicts = []

    for setting in settings:
        setting = add_missing_defaults(setting)
        devices = setting["device"]

        for dev in devices:
            problem = copy.deepcopy(setting)
            problem["device"] = dev
            problem_dicts.append(problem)

    return [ExtensionsTestProblem(**p) for p in problem_dicts]


def add_missing_defaults(setting):
    """Create full settings from setting.

    Args:
        setting (dict): configuration dictionary

    Returns:
        dict: full settings.

    Raises:
        ValueError: if no proper settings
    """
    required = ["module_fn", "input_fn", "loss_function_fn", "target_fn"]
    optional = {
        "id_prefix": "",
        "seed": 0,
        "device": get_available_devices(),
    }

    for req in required:
        if req not in setting.keys():
            raise ValueError("Missing configuration entry for {}".format(req))

    for opt, default in optional.items():
        if opt not in setting.keys():
            setting[opt] = default

    for s in setting.keys():
        if s not in required and s not in optional.keys():
            raise ValueError("Unknown config: {}".format(s))

    return setting


class ExtensionsTestProblem:
    """Class providing functions and parameters."""

    def __init__(
        self,
        input_fn,
        module_fn,
        loss_function_fn,
        target_fn,
        device,
        seed,
        id_prefix,
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
        """Set up problem from settings."""
        torch.manual_seed(self.seed)

        self.model = self.module_fn().to(self.device)
        self.input = self.input_fn().to(self.device)
        self.target = self.target_fn().to(self.device)
        self.loss_function = self.loss_function_fn().to(self.device)

    def tear_down(self):
        """Delete all variables after problem."""
        del self.model, self.input, self.target, self.loss_function

    def make_id(self):
        """Needs to function without call to `set_up`.

        Returns:
            str: id of problem
        """
        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""
        return (
            prefix
            + "dev={}-in={}-model={}-loss={}".format(
                self.device,
                tuple(self.input_fn().shape),
                self.module_fn(),
                self.loss_function_fn(),
            ).replace(" ", "")
        )

    def forward_pass(self, sample_idx=None):
        """Do a forward pass. Return input, output, and parameters.

        The forward pass is performed on the selected index.
        If the index is None, then the forward pass is calculated for the whole batch.

        Args:
            sample_idx (int, optional): Index of the sample to select.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                input, output, loss, each with batch axis first
        """
        if sample_idx is None:
            input = self.input.clone().detach()
            target = self.target.clone().detach()
        else:
            target = self.target.split(1, dim=0)[sample_idx].detach()
            input = self.input.split(1, dim=0)[sample_idx].detach()

        output = self.model(input)

        loss = self.loss_function(output, target)

        return input, output, loss

    def extend(self):
        """Extend module of problem."""
        self.model = extend(self.model)
        self.loss_function = extend(self.loss_function)

    def get_reduction_factor(self, loss, unreduced_loss):
        """Return the factor used to reduce the individual losses.

        Args:
            loss (torch.Tensor): the loss after reduction
            unreduced_loss (torch.Tensor): the raw loss before reduction

        Returns:
            float: factor

        Raises:
            RuntimeError: if either mean or sum cannot be determined
        """
        mean_loss = unreduced_loss.flatten().mean()
        sum_loss = unreduced_loss.flatten().sum()
        if torch.allclose(mean_loss, sum_loss):
            raise RuntimeError(
                "Cannot determine reduction factor. ",
                "Results from 'mean' and 'sum' reduction are identical. ",
                f"'mean': {mean_loss}, 'sum': {sum_loss}",
            )
        if torch.allclose(loss, mean_loss):
            factor = 1.0 / unreduced_loss.numel()
        elif torch.allclose(loss, sum_loss):
            factor = 1.0
        else:
            raise RuntimeError(
                "Reductions 'mean' or 'sum' do not match with loss. ",
                f"'mean': {mean_loss}, 'sum': {sum_loss}, loss: {loss}",
            )
        return factor

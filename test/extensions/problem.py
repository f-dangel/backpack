"""Convert problem settings."""

import copy
from test.core.derivatives.utils import get_available_devices

import torch

from backpack import extend


def make_test_problems(settings):
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
    """Create extensions test problem from setting.
    Args:
        setting (dict): configuration dictionary
    Returns:
        ExtensionsTestProblem: problem with specified settings.
    """
    required = ["module_fn", "input_fn", "loss_function_fn", "target_fn"]
    optional = {
        "id_prefix": "",
        "seed": 0,
        "device": get_available_devices(),
        "axis_batch": 0,
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
    def __init__(
        self,
        input_fn,
        module_fn,
        loss_function_fn,
        target_fn,
        device,
        seed,
        id_prefix,
        axis_batch,
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
            axis_batch (int): index of batch axis. Defaults to 0.
        """
        self.module_fn = module_fn
        self.input_fn = input_fn
        self.loss_function_fn = loss_function_fn
        self.target_fn = target_fn

        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix
        self.axis_batch = axis_batch

    def set_up(self):
        torch.manual_seed(self.seed)

        self.model = self.module_fn().to(self.device)
        self.input = self.input_fn().to(self.device)
        self.target = self.target_fn().to(self.device)
        self.loss_function = self.loss_function_fn().to(self.device)

    def tear_down(self):
        del self.model, self.input, self.target, self.loss_function

    def make_id(self):
        """Needs to function without call to `set_up`."""
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
        """Do a forward pass. Return input, output, and parameters."""
        if sample_idx is None:
            input = self.input.clone().detach()
            target = self.target.clone().detach()
        else:
            target = self.target.clone()[sample_idx].unsqueeze(0).detach()
            input = torch.split(
                self.input, split_size_or_sections=1, dim=self.axis_batch
            )[sample_idx].detach()

        # TODO double check this structure,
        # maybe forward() and then self.output is more appropriate
        if isinstance(self.model, torch.nn.RNN):
            output, _ = self.model(input)
        else:
            output = self.model(input)
        output = output.swapaxes(0, self.axis_batch)

        loss = self.loss_function(output, target)

        return input, output, loss

    def extend(self):
        self.model = extend(self.model)
        self.loss_function = extend(self.loss_function)

    def get_reduction_factor(self, loss, unreduced_loss):
        """Return the factor used to reduce the individual losses."""
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

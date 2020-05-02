"""Convert problem settings."""

import copy
from test.core.derivatives.utils import (
    derivative_cls_for,
    get_available_devices,
    is_loss,
)

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

    return [DerivativesTestProblem(**p) for p in problem_dicts]


def add_missing_defaults(setting):
    """Create derivative test problem from setting.

    Args:
        setting (dict): configuration dictionary

    Returns:
        DerivativesTestProblem: problem with specified settings.
    """
    required = ["module_fn", "module_kwargs", "input_kwargs"]
    optional = {
        "input_fn": torch.rand,
        "id_prefix": "",
        "seed": 0,
        "target_fn": lambda: None,
        "target_kwargs": {},
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


class DerivativesTestProblem:
    def __init__(
        self,
        module_fn,
        module_kwargs,
        input_fn,
        input_kwargs,
        target_fn,
        target_kwargs,
        device,
        seed,
        id_prefix,
    ):
        """Collection of information required to test derivatives.

        Args:
            module (torch.nn.Module): Layer to be tested.
            input_shape (tuple(int)): Layer input shape with batch size
            device (torch.Device): Layer device (CPU, GPU, ...)
            id_prefix (str): Prefix used for test ID

        Notes:
            Shape of module input → output:
            [N, C_in, H_in, W_in, ...] → [N, C_out, H_out, W_out, ...]
        """
        torch.manual_seed(seed)

        self.module = module_fn(**module_kwargs).to(device)

        print(input_fn)
        print(input_kwargs)
        self.input = input_fn(**input_kwargs).to(device)
        self.target = target_fn(**target_kwargs)
        if self.target is not None:
            self.target = self.target.to(device)

        self.derivative = derivative_cls_for(self.module.__class__)()

        self.input_shape = self.input.shape
        self.output_shape = self.get_output_shape()

        self.device = device

        self.id_prefix = id_prefix

    def is_loss(self):
        return is_loss(self.module)

    def forward_pass(self, input_requires_grad=False):
        """Do a forward pass. Return input, output, and parameters."""
        input = self.input
        if input_requires_grad:
            input.requires_grad = True

        if self.is_loss():
            output = self.module(input, self.target)
        else:
            output = self.module(input)

        return input, output, dict(self.module.named_parameters())

    def get_output_shape(self):
        _, output, _ = self.forward_pass()
        return tuple(output.shape)

    def make_id(self):
        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""
        return prefix + "dev={}-in={}-{}".format(
            self.device, tuple(self.input_shape), self.module
        ).replace(" ", "")

    def extend(self):
        self.module = extend(self.module)

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
    """Add missing entries in settings such that the new format works."""
    required = ["module_fn", "input_fn"]
    optional = {
        "target_fn": lambda: None,
        "id_prefix": "",
        "device": get_available_devices(),
        "seed": 0,
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
        input_fn,
        target_fn,
        device,
        seed,
        id_prefix,
    ):
        """Collection of information required to test derivatives.

        Warning:
            Initialization is lazy. `set_up` needs to be called before the
            test problem can be used

        Args:
            module (torch.nn.Module): Layer to be tested.
            input_shape (tuple(int)): Layer input shape with batch size
            device (torch.Device): Layer device (CPU, GPU, ...)
            id_prefix (str): Prefix used for test ID

        Notes:
            Shape of module input → output:
            [N, C_in, H_in, W_in, ...] → [N, C_out, H_out, W_out, ...]
        """
        self.module_fn = module_fn
        self.input_fn = input_fn
        self.target_fn = target_fn
        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix

    def set_up(self):
        torch.manual_seed(self.seed)

        self.module = self.make_module()
        self.input = self.make_input()
        self.target = self.make_target()
        self.derivative = self.make_derivative()
        self.input_shape = self.make_input_shape()
        self.output_shape = self.make_output_shape()

    def tear_down(self):
        del self.module
        del self.input
        del self.target
        del self.derivative
        del self.input_shape
        del self.output_shape

    def make_module(self):
        return self.module_fn().to(self.device)

    def make_input(self):
        return self.input_fn().to(self.device)

    def make_target(self):
        target = self.target_fn()
        if target is not None:
            target = target.to(self.device)
        return target

    def make_derivative(self):
        module = self.make_module()
        return derivative_cls_for(module.__class__)()

    def make_input_shape(self):
        return self.make_input().shape

    def make_output_shape(self):
        module = self.make_module()
        input = self.make_input()
        target = self.make_target()

        if target is None:
            output = module(input)
        else:
            output = module(input, target)

        return output.shape

    def is_loss(self):
        return is_loss(self.make_module())

    def forward_pass(self, input_requires_grad=False, sample_idx=None):
        """Do a forward pass. Return input, output, and parameters."""
        if sample_idx is None:
            input = self.input.clone().detach()
        else:
            input = self.input.clone()[sample_idx, :].unsqueeze(0).detach()

        if input_requires_grad:
            input.requires_grad = True

        if self.is_loss():
            assert sample_idx is None
            output = self.module(input, self.target)
        else:
            output = self.module(input)

        return input, output, dict(self.module.named_parameters())

    def make_id(self):
        """Needs to function without call to `set_up`."""
        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""
        return (
            prefix
            + "dev={}-in={}-{}".format(
                self.device,
                tuple(self.make_input_shape()),
                self.make_module(),
            ).replace(" ", "")
        )

    def extend(self):
        self.module = extend(self.module)

    def has_weight(self):
        module = self.make_module()
        return hasattr(module, "weight") and module.weight is not None

    def has_bias(self):
        module = self.make_module()
        return hasattr(module, "bias") and module.bias is not None

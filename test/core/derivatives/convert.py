"""Convert problem settings."""

import copy
from test.core.derivatives.utils import derivative_cls_for, get_available_devices

import torch

from backpack import extend


def make_test_problems(settings):
    individual_settings = make_individual_settings(settings)

    return [DerivativesTestProblem.from_setting(s) for s in individual_settings]


def make_individual_settings(settings):
    individual_settings = []

    for setting in settings:
        if "device" not in setting.keys():
            setting["device"] = get_available_devices()

    for setting in settings:
        if isinstance(setting["device"], list):
            for dev in setting["device"]:
                individual_setting = copy.deepcopy(setting)
                individual_setting["device"] = dev
                individual_settings.append(individual_setting)

    return individual_settings


class DerivativesTestProblem:
    def __init__(self, module, in_shape, device, id_prefix=None):
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
        self.module = module.to(device)
        self.derivative = derivative_cls_for(module.__class__)()
        self.in_shape = in_shape
        torch.manual_seed(123)
        self.input = torch.rand(*in_shape).to(device)
        self.device = device
        self.id_prefix = "" if id_prefix is None else id_prefix
        self.out_shape = self.get_output_shape()

    def get_output_shape(self):
        output = self.module(self.input)
        return tuple(output.shape)

    def make_id(self):
        prefix = (self.id_prefix + "-") if self.id_prefix is not None else ""
        return prefix + "dev={}-in={}-{}".format(
            self.device, self.in_shape, self.module
        ).replace(" ", "")

    def extend(self):
        self.module = extend(self.module)

    @classmethod
    def from_setting(cls, setting):
        """Create derivative test problem from setting.

        Args:
            setting (dict): configuration dictionary

        Returns:
            DerivativesTestProblem: problem with specified settings.
        """
        required = ["module_cls", "in_shape", "device"]
        optional = {"id_prefix": None, "module_kwargs": {}}

        for req in required:
            if req not in setting.keys():
                raise ValueError("Missing configuration entry for {}".format(req))

        for opt, default in optional.items():
            if opt not in setting.keys():
                setting[opt] = default

        module_cls = setting["module_cls"]
        module_kwargs = setting["module_kwargs"]
        module = module_cls(**module_kwargs)

        in_shape = setting["in_shape"]
        device = setting["device"]
        id_prefix = setting["id_prefix"]

        return cls(module, in_shape, device, id_prefix=id_prefix)

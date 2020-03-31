"""Test class for module partial derivatives.

- Jacobian-matrix products
- transposed Jacobian-matrix products
"""

import warnings

import pytest
import torch

# TODO: Move to a utils file, fix imports
from ...automated_test import check_sizes, check_values


class DerivativesModule:
    """ Information required to test a class inheriting from
    `backpack.core.derivatives.BaseDerivatives`.

    batch_size : N
    input_shape : [C_in, H_in, W_in, ...]

    Shape of module input → output:
      [N, C_in, H_in, W_in, ...] → [N, C_out, H_out, W_out, ...]

    """

    def __init__(self, module, N, input_shape, device):
        self.module = module
        # TODO: write functionality to get derivative class
        # from module using backpack.core.derivatives.derivatives_for dict
        self.derivative = None
        self.N = N
        self.input_shape = input_shape
        self.device = device


class DerivativesImplementation:
    """Base class for autograd and BackPACK implementations."""

    def __init__(self, derivatives_module):
        self.derivatives_module = derivatives_module

    def jac_t_mat_prod(self, mat):
        raise NotImplementedError


class BackpackDerivativesImplementation(DerivativesImplementation):
    """Derivative implementations with BackPACK."""

    DUMMY = torch.zeros(1)

    def jac_t_mat_prod(self, mat):
        warnings.warn("Using dummy Backpack implementation.")
        return self.DUMMY


class AutogradDerivativesImplementation(DerivativesImplementation):
    """Derivative implementations with autograd.

    """

    DUMMY = torch.zeros(1)

    # TODO: Call transposed Jacobian from backpack.hessianfree
    def jac_t_mat_prod(self, mat):
        warnings.warn("Using dummy Autograd implementation.")
        return self.DUMMY


# TODO: Move to a utils file
def get_available_devices():
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.add(torch.device("cuda"))

    return devices


def set_up_derivatives_module(setup, device):
    """Create a DerivativesModule object from setup."""
    module_cls = setup["module_cls"]
    module_kwargs = setup["module_kwargs"]
    N = setup["N"]
    input_shape = setup["input_shape"]

    module = module_cls(**module_kwargs)
    derivatives_module_id = "{}-{}-N={}-input_shape={}".format(
        device, module, N, input_shape
    )

    return DerivativesModule(module, N, input_shape, device), derivatives_module_id


##############################################################################
# TODO: Move to test_linear.py
# In general, the files test_sigmoid, test_avgpool2d, ... should contain such
# a list, which is merged in this file

# Setting up modules for the test, not sure if this is the best way
SETUPS = [
    # first setup
    {
        "module_cls": torch.nn.Linear,
        "module_kwargs": {"in_features": 7, "out_features": 3, "bias": True,},
        "input_shape": (7,),
        "N": 10,
    },
    # second setup
    {
        "module_cls": torch.nn.Linear,
        "module_kwargs": {"in_features": 10, "out_features": 1, "bias": False,},
        "input_shape": (10,),
        "N": 1,
    },
]
##############################################################################

##############################################################################
# create test for each device
DEVICES = get_available_devices()
# TODO: Convert to function
ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []

for setup in SETUPS:
    for device in DEVICES:
        derivatives_module, derivatives_module_id = set_up_derivatives_module(
            setup, device
        )

        ALL_CONFIGURATIONS.append(derivatives_module)
        CONFIGURATION_IDS.append(derivatives_module_id)

##############################################################################


@pytest.mark.parametrize(
    "derivatives_module", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS
)
def test_jac_t_mat_prod(derivatives_module, V=10):
    """Test the transposed Jacobian-matrix product. """
    # TODO: Set up matrix of size [V, N, C_in, H_in, W_in]
    warnings.warn("This is a dummy.")
    mat = None

    backpack_res = BackpackDerivativesImplementation(derivatives_module).jac_t_mat_prod(
        mat
    )
    autograd_res = AutogradDerivativesImplementation(derivatives_module).jac_t_mat_prod(
        mat
    )

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)

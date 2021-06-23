"""Test problems for the AdaptiveAvgPool shape checker."""
from __future__ import annotations

import copy
from test.automated_test import check_sizes_and_values
from test.core.derivatives.utils import get_available_devices
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor, randn
from torch.nn import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Module,
)

from backpack import extend
from backpack.core.derivatives.adaptive_avg_pool_nd import AdaptiveAvgPoolNDDerivatives


def make_test_problems(settings: List[Dict[str, Any]]) -> List[AdaptiveAvgPoolProblem]:
    """Creates the test problem from settings.

    Args:
        settings: list of dictionaries with settings

    Returns:
        a list of the test problems
    """
    problem_dicts: List[Dict[str, Any]] = []

    for setting in settings:
        setting = add_missing_defaults(setting)
        devices = setting["device"]

        for dev in devices:
            problem = copy.deepcopy(setting)
            problem["device"] = dev
            problem_dicts.append(problem)

    return [AdaptiveAvgPoolProblem(**p) for p in problem_dicts]


def add_missing_defaults(setting: Dict[str, Any]) -> Dict[str, Any]:
    """Add missing entries in settings such that the new format works.

    Args:
        setting: dictionary with required settings and some optional settings

    Returns:
        complete settings including the default values for missing optional settings

    Raises:
        ValueError: if the settings do not work
    """
    required = ["N", "shape_input", "shape_target", "works"]
    optional = {
        "id_prefix": "",
        "device": get_available_devices(),
        "seed": 0,
    }

    for req in required:
        if req not in setting.keys():
            raise ValueError(f"Missing configuration entry for {req}")

    for opt, default in optional.items():
        if opt not in setting.keys():
            setting[opt] = default

    for s in setting.keys():
        if s not in required and s not in optional.keys():
            raise ValueError(f"Unknown config: {s}")

    return setting


class AdaptiveAvgPoolProblem:
    """Test problem for testing AdaptiveAvgPoolNDDerivatives.check_parameters()."""

    def __init__(
        self,
        N: int,
        shape_input: Any,
        shape_target: Tuple[int],
        works: bool,
        device,
        seed: int,
        id_prefix: str,
    ):
        """Initialization.

        Args:
            N: number of dimensions
            shape_input: input shape
            shape_target: target shape
            works: whether the test should run without errors
            device: device
            seed: seed for torch
            id_prefix: prefix for problem id

        Raises:
            NotImplementedError: if N is not in [1, 2, 3]
        """
        if N not in [1, 2, 3]:
            raise NotImplementedError(f"N={N} not implemented in test suite.")
        self.N = N
        self.shape_input = shape_input
        self.shape_target = shape_target
        self.works = works
        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix

    def make_id(self) -> str:
        """Create an id from problem parameters.

        Returns:
            problem id
        """
        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""
        return (
            prefix + f"dev={self.device}-N={self.N}-in={self.shape_input}-"
            f"out={self.shape_target}-works={self.works}"
        )

    def set_up(self) -> None:
        """Set up problem and do one forward pass."""
        torch.manual_seed(self.seed)
        self.module = self._make_module()
        self.input = randn(self.shape_input)
        self.output = self.module(self.input)

    def tear_down(self):
        """Delete created torch variables."""
        del self.module
        del self.input
        del self.output

    def _make_module(
        self,
    ) -> Union[AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d]:
        map_class = {1: AdaptiveAvgPool1d, 2: AdaptiveAvgPool2d, 3: AdaptiveAvgPool3d}
        module = map_class[self.N](output_size=self.shape_target)
        return extend(module.to(device=self.device))

    def check_parameters(self) -> None:
        """Key method for test.

        Run the AdaptiveAvgPoolNDDerivatives.check_parameters() method.
        """
        self._get_derivatives().check_parameters(module=self.module)

    def _get_derivatives(self) -> AdaptiveAvgPoolNDDerivatives:
        return AdaptiveAvgPoolNDDerivatives(N=self.N)

    def check_equivalence(self) -> None:
        """Check if the given parameters lead to the same output.

        Checks the sizes and values.
        """
        stride, kernel_size, _ = self._get_derivatives().get_avg_pool_parameters(
            self.module
        )
        module_equivalent: Module = self._make_module_equivalent(stride, kernel_size)
        output_equivalent: Tensor = module_equivalent(self.input)

        check_sizes_and_values(self.output, output_equivalent)

    def _make_module_equivalent(
        self, stride: List[int], kernel_size: List[int]
    ) -> Union[AvgPool1d, AvgPool2d, AvgPool3d]:
        map_class = {1: AvgPool1d, 2: AvgPool2d, 3: AvgPool3d}
        module = map_class[self.N](kernel_size=kernel_size, stride=stride)
        return module.to(self.device)

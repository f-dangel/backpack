"""Convert problem settings."""

import copy
from test.core.derivatives.utils import get_available_devices
from typing import Any, Iterator, List, Tuple

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from backpack import extend
from backpack.utils.subsampling import subsample


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
        return prefix + "dev={}-in={}-model={}-loss={}".format(
            self.device,
            tuple(self.input_fn().shape),
            self.module_fn(),
            self.loss_function_fn(),
        ).replace(" ", "")

    def forward_pass(
        self, subsampling: List[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Do a forward pass. Return input, output, and parameters.

        If sub-sampling is None, the forward pass is calculated on the whole batch.

        Args:
            subsampling: Indices of selected samples. Default: ``None`` (all samples).

        Returns:
            input, output, and loss of the forward pass
        """
        input = self.input.clone()
        target = self.target.clone()

        if subsampling is not None:
            batch_axis = 0
            input = subsample(self.input, dim=batch_axis, subsampling=subsampling)
            target = subsample(self.target, dim=batch_axis, subsampling=subsampling)

        output = self.model(input)
        loss = self.loss_function(output, target)

        return input, output, loss

    def extend(self):
        """Extend module of problem."""
        self.model = extend(self.model)
        self.loss_function = extend(self.loss_function)

    @staticmethod
    def __get_reduction_factor(loss: Tensor, unreduced_loss: Tensor) -> float:
        """Return the factor used to reduce the individual losses.

        Args:
            loss: Reduced loss.
            unreduced_loss: Unreduced loss.

        Returns:
            Reduction factor.

        Raises:
            RuntimeError: if either mean or sum cannot be determined
        """
        mean_loss = unreduced_loss.flatten().mean()
        sum_loss = unreduced_loss.flatten().sum()
        if torch.allclose(mean_loss, sum_loss):
            if unreduced_loss.numel() == 1 and torch.allclose(loss, sum_loss):
                factor = 1.0
            else:
                raise RuntimeError(
                    "Cannot determine reduction factor. ",
                    "Results from 'mean' and 'sum' reduction are identical. ",
                    f"'mean': {mean_loss}, 'sum': {sum_loss}",
                )
        elif torch.allclose(loss, mean_loss):
            factor = 1.0 / unreduced_loss.numel()
        elif torch.allclose(loss, sum_loss):
            factor = 1.0
        else:
            raise RuntimeError(
                "Reductions 'mean' or 'sum' do not match with loss. ",
                f"'mean': {mean_loss}, 'sum': {sum_loss}, loss: {loss}",
            )
        return factor

    def trainable_parameters(self) -> Iterator[Parameter]:
        """Yield the model's trainable parameters.

        Yields:
            Model parameter with gradients enabled.
        """
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def collect_data(self, savefield: str) -> List[Any]:
        """Collect BackPACK attributes from trainable parameters.

        Args:
            savefield: Attribute name.

        Returns:
            List of attributes saved under the trainable model parameters.

        Raises:
            RuntimeError: If a non-differentiable parameter with the attribute is
                encountered.
        """
        data = []

        for p in self.model.parameters():
            if p.requires_grad:
                data.append(getattr(p, savefield))
            else:
                if hasattr(p, savefield):
                    raise RuntimeError(
                        f"Found non-differentiable parameter with attribute '{savefield}'."
                    )

        return data

    def get_batch_size(self) -> int:
        """Return the mini-batch size.

        Returns:
            Mini-batch size.
        """
        return self.input.shape[0]

    def compute_reduction_factor(self) -> float:
        """Compute loss function's reduction factor for aggregating per-sample losses.

        For instance, if ``reduction='mean'`` is used, then the reduction factor
        is ``1 / N`` where ``N`` is the batch size. With ``reduction='sum'``, it
        is ``1``.

        Returns:
            Reduction factor
        """
        _, _, loss = self.forward_pass()

        batch_size = self.get_batch_size()
        loss_list = torch.zeros(batch_size, device=self.device)

        for n in range(batch_size):
            _, _, loss_n = self.forward_pass(subsampling=[n])
            loss_list[n] = loss_n

        return self.__get_reduction_factor(loss, loss_list)

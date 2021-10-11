import itertools

import pytest
import torch

import backpack
from backpack.core.derivatives.convnd import weight_jac_t_save_memory


def parameters_issue_30():
    possible_values = {
        "N": [4],
        "C_in": [4],
        "C_out": [6],
        "H": [6],
        "W": [6],
        "K": [3],
        "S": [1, 3],
        "pad": [0, 2],
        "dil": [1, 2],
    }

    configs = [
        dict(zip(possible_values.keys(), config_tuple))
        for config_tuple in itertools.product(*possible_values.values())
    ]

    return {
        "argvalues": configs,
        "ids": [str(config) for config in configs],
    }


@pytest.mark.parametrize("params", **parameters_issue_30())
@pytest.mark.parametrize(
    "save_memory",
    [True, False],
    ids=["save_memory=True", "save_memory=False"],
)
def test_convolutions_stride_issue_30(params, save_memory):
    """
    https://github.com/f-dangel/backpack/issues/30

    The gradient for the convolution is wrong when `stride` is not a multiple of
    `D + 2*padding - dilation*(kernel-1) - 1`.
    """
    torch.manual_seed(0)

    mod = torch.nn.Conv2d(
        in_channels=params["C_in"],
        out_channels=params["C_out"],
        kernel_size=params["K"],
        stride=params["S"],
        padding=params["pad"],
        dilation=params["dil"],
    )
    backpack.extend(mod)
    x = torch.randn(size=(params["N"], params["C_in"], params["W"], params["H"]))

    with weight_jac_t_save_memory(save_memory), backpack.backpack(
        backpack.extensions.BatchGrad()
    ):
        loss = torch.sum(mod(x))
        loss.backward()

        for p in mod.parameters():
            assert torch.allclose(p.grad, p.grad_batch.sum(0), rtol=1e-04, atol=1e-04)

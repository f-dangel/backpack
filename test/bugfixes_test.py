import itertools

import pytest
import torch

import backpack


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
def test_convolutions_stride_issue_30(params):
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

    with backpack.backpack(backpack.extensions.BatchGrad()):
        loss = torch.sum(mod(x))
        loss.backward()

        for p in mod.parameters():
            assert torch.allclose(p.grad, p.grad_batch.sum(0), rtol=1e-04, atol=1e-04)

"""Individual Hessian-Vector-Product and retain_graph
======================================================
"""
from typing import List

import torch
from torch import Tensor, nn, zeros
from torch.nn import Module

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.examples import load_one_batch_mnist


def batch_hvp(
    model: Module, loss_func: Module, x: Tensor, y: Tensor, v_list: List[Tensor]
) -> List[Tensor]:
    """Compute individual Hessian-vector products."""
    param_list = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad()

    with backpack(retain_graph=True):
        loss = loss_func(model(x), y)
        loss.backward(retain_graph=True, create_graph=True)

    dot_product = sum((v * p.grad).flatten().sum() for v, p in zip(v_list, param_list))

    with backpack(BatchGrad()):
        dot_product.backward()

    return [p.grad_batch for p in param_list]


def batch_hvp_for_loop(
    model: Module, loss_func: Module, x: Tensor, y: Tensor, v_list: List[Tensor]
) -> List[Tensor]:
    """Compute individual Hessian-vector products with autograd."""
    param_list = [p for p in model.parameters() if p.requires_grad]
    N = x.shape[0]
    batch_hvp = [zeros(N, *p.shape, dtype=p.dtype, device=p.device) for p in param_list]

    for n in range(N):
        x_n, y_n = x[[n]], y[[n]]
        loss_n = loss_func(model(x_n), y_n)

        hvp_n = hessian_vector_product(loss_n, param_list, v_list)

        for param_idx, param_hvp_n in enumerate(hvp_n):
            batch_hvp[param_idx][n] = param_hvp_n

    return batch_hvp


def make_working_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def make_failing_model():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 10, bias=True), nn.Linear(10, 10, bias=True)
    )


def compare_batch_hvp(model_fn, print_mismatches: bool = False):
    """Check whether results of ``batch_hvp`` and ``batch_hvp_for_loop`` match."""
    N = 1
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = extend(model_fn().to(DEVICE))
    loss_function = extend(nn.CrossEntropyLoss(reduction="sum").to(DEVICE))

    x, y = load_one_batch_mnist(N)
    x, y = x.to(DEVICE), y.to(DEVICE)

    v_list = [torch.rand_like(p) for p in model.parameters() if p.requires_grad]

    ihvp = batch_hvp(model, loss_function, x, y, v_list)
    ihvp_for_loop = batch_hvp_for_loop(model, loss_function, x, y, v_list)

    fail = False

    for result, result_compare in zip(ihvp, ihvp_for_loop):
        rtol = 1e-3
        atol = 1e-5
        match = torch.allclose(result, result_compare, rtol=rtol, atol=atol)
        print(f"IHVPs match: {match}")

        if match is False:
            fail = True

        if match is False and print_mismatches:
            for el1, el2 in zip(result.flatten(), result_compare.flatten()):
                if not torch.allclose(el1, el2, rtol=rtol, atol=atol):
                    print(f"{el1} versus {el2}")

    if fail:
        raise Exception("At least one IHVP does not match.")


if __name__ == "__main__":
    print("\nWorking model (parameters appear once in graph)")
    compare_batch_hvp(make_working_model)
    print("\nFailing model (weight of 2nd layer appears multiple times in graph)")
    compare_batch_hvp(make_failing_model)

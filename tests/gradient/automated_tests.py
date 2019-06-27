"""Testing class for comparison of first-order information and brute-force
 auto-differentiation."""

import torch
from bpexts.utils import set_seeds
from .test_problem import TestProblem, losses
from .test_problem_checker import unittest_for


def make_test_problem(layer_fn, input_size, loss, device, seed=0):
    set_seeds(seed)
    model = layer_fn()
    set_seeds(seed)
    X = torch.randn(input_size)
    Y = 1 - model(X)
    return TestProblem(X, Y, model, loss, device=device)


def set_up_tests(layer_fn, layer_name, input_size, atol=1e-8, rtol=1e-5):
    """Yield the names and classes for the unittests."""

    for idx, loss in enumerate(losses):
        problem = make_test_problem(
            layer_fn,
            input_size,
            loss,
            device=torch.device('cpu')
        )
        cpu_test = unittest_for(
            problem,
            atol=atol,
            rtol=rtol
        )

        yield '{}CPUTest{}'.format(layer_name, idx), cpu_test

        if torch.cuda.is_available():
            problem = make_test_problem(
                layer_fn,
                input_size,
                loss,
                device=torch.device('gpu:0')
            )

            gpu_test = unittest_for(
                problem,
                atol=atol,
                rtol=rtol
            )

            yield '{}GPUTest{}'.format(layer_name, idx), gpu_test

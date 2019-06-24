"""Testing class for comparison of first-order information and brute-force
 auto-differentiation."""

import torch
import numpy
import unittest
from bpexts.utils import set_seeds
import bpexts.gradient.config as config


def gradient_test(layer_fn, input_size, device, seed=0, atol=1e-5, rtol=1e-8):
    """Create unittest for checking first-order information via auto-diff.

    Checks for the following aspects:
    - batchwise gradients
    - TODO sum of gradients squared

    Parameters:
    -----------
    layer_fn : function
        Returns the layer that will be checked
    input_size : tuple(int)
        Dimension of the input (with batch dimension)
    device : torch.device
        Device to run the tests on
    atol : float
        Absolute tolerance for elementwise comparison
    rtol : float
        Relative tolerance for elementwise comparison

    Returns:
    --------
    unittest.TestCase
        test case for comparing the first-order functionalities with autograd
    """

    class GradientTest0(unittest.TestCase):
        SEED = seed
        DEVICE = device
        INPUT_SIZE = input_size
        ATOL = atol
        RTOL = rtol

        def _create_layer(self):
            return layer_fn().to(self.DEVICE)

        def _loss_fn(self, x):
            """Dummy loss function: Normalized sum of squared elements.

            The loss Hessian of this function is constant and diagonal.
            """
            return (x**2).contiguous().view(x.size(0), -1).mean(0).sum()

        def test_batch_gradients(self):
            """Check for same forward pass."""
            # create input
            autograd_res = self._compute_batch_gradients_autograd()
            bpexts_res = self._compute_batch_gradients_bpexts()
            assert len(autograd_res) == len(bpexts_res)
            for g1, g2 in zip(autograd_res, bpexts_res):
                assert g1.size() == g2.size()
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def _compute_batch_gradients_autograd(self):
            layer = self._create_layer()
            inputs = self._create_input()
            batch_grads = [
                torch.zeros(inputs.size(0), *p.size()).to(self.DEVICE)
                for p in layer.parameters()
            ]
            for b in range(inputs.size(0)):
                sample = inputs[b, :].unsqueeze(0)
                loss = self._loss_fn(layer(sample))
                gradients = torch.autograd.grad(loss, layer.parameters())
                for idx, g in enumerate(gradients):
                    batch_grads[idx][b, :] = g / inputs.size(0)
            return batch_grads

        def _compute_batch_gradients_bpexts(self):
            layer = self._create_layer()
            inputs = self._create_input()
            loss = self._loss_fn(layer(inputs))
            with config.bpexts(config.BATCH_GRAD):
                loss.backward()
                batch_grads = [p.grad_batch for p in layer.parameters()]
                layer.zero_grad()
                layer.clear_grad_batch()
            return batch_grads

        def _create_input(self):
            """Return same random input (reset seed before)."""
            set_seeds(self.SEED)
            return torch.randn(*self.INPUT_SIZE).to(self.DEVICE)

    class GradientTest1(GradientTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Normalized squared sum."""
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += (x[b, :].view(-1).sum())**2 / x.size(0)
            return loss

    class GradientTest2(GradientTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Sum of log10 of shifted normalized abs.

            The loss Hessian of this function is non-constant and non-diagonal.
            """
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += (
                    torch.log10(torch.abs(x[b, :]) + 0.1).sum())**2 / x.size(0)
            return loss

    return GradientTest0, GradientTest1, GradientTest2


def set_up_gradient_tests(layer_fn,
                          layer_name,
                          input_size,
                          atol=1e-8,
                          rtol=1e-5):
    """Yield the names and classes for the unittests."""
    # create CPU tests
    for idx, cpu_test in enumerate(
            gradient_test(
                layer_fn,
                input_size,
                device=torch.device('cpu'),
                atol=atol,
                rtol=rtol)):
        print(cpu_test.DEVICE)
        yield '{}CPUTest{}'.format(layer_name, idx), cpu_test

    # create GPU tests if available
    if torch.cuda.is_available():
        for idx, gpu_test in enumerate(
                gradient_test(
                    layer_fn,
                    input_size,
                    device=torch.device('cuda:0'),
                    atol=atol,
                    rtol=rtol)):
            print(gpu_test.DEVICE)
            yield '{}GPUTest{}'.format(layer_name, idx), gpu_test

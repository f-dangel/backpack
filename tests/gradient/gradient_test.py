"""Testing class for comparison of first-order information and brute-force
 auto-differentiation."""

import torch
import pytest
import numpy
import unittest
from bpexts.utils import set_seeds
from .test_problem import TestProblem


def loss0(x, y=None):
    """Dummy loss function: Normalized sum of squared elements."""
    return (x**2).contiguous().view(x.size(0), -1).mean(0).sum()


def loss1(x, y=None):
    loss = torch.zeros(1).to(x.device)
    for b in range(x.size(0)):
        loss += (x[b, :].view(-1).sum())**2 / x.size(0)
    return loss


def loss2(x, y=None):
    loss = torch.zeros(1).to(x.device)
    for b in range(x.size(0)):
        loss += (
            torch.log10(torch.abs(x[b, :]) + 0.1).sum())**2 / x.size(0)
    return loss


def make_test_problem(seed, layer_fn, input_size, loss, device):
    set_seeds(seed)
    model = layer_fn()
    set_seeds(seed)
    X = torch.randn(input_size)
    Y = 1 - model(X)
    return TestProblem(X, Y, model, loss, device=device)


def gradient_test(layer_fn, input_size, device, seed=0, atol=1e-5, rtol=1e-8):
    """Create unittest for checking first-order information via auto-diff.

    Checks for the following aspects:
    - batchwise gradients
    - sum of gradients squared

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
        TEST_SUM_GRAD_SQUARED = True
        TEST_DIAG_GGN = False

        test_problem = make_test_problem(seed, layer_fn, input_size, loss0, device)

        def _loss_fn(self, x):
            """Dummy loss function: Normalized sum of squared elements."""
            return loss0(x)

        def test_batch_gradients(self):
            """Check for same batch gradients."""
            # create input
            autograd_res = self.test_problem.batch_gradients_autograd()
            bpexts_res = self.test_problem.batch_gradients_bpexts()
            layer = self.test_problem.model
            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for g1, g2, p in zip(autograd_res, bpexts_res, layer.parameters()):
                assert tuple(g1.size()) == tuple(
                    g2.size()) == (self.INPUT_SIZE[0], ) + tuple(p.size())
                self._residuum_report(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_batch_gradients_sum_to_grad(self):
            """Check if the batch sum of gradients yields the gradient."""
            layer = self.test_problem.model
            autograd_gradients = self.test_problem.gradient_autograd()
            bpexts_batch_gradients = self.test_problem.batch_gradients_bpexts()
            assert len(autograd_gradients) == len(
                bpexts_batch_gradients) == len(list(layer.parameters()))
            for g, batch_g, p in zip(
                    autograd_gradients, bpexts_batch_gradients,
                    layer.parameters()):
                bpexts_g = batch_g.sum(0)
                assert g.size() == bpexts_g.size() == p.size()
                self._residuum_report(g, bpexts_g)
                assert torch.allclose(
                    g, bpexts_g, atol=self.ATOL, rtol=self.RTOL)

        def test_sgs(self):
            """Check for same sum of gradients squared."""
            if self.TEST_SUM_GRAD_SQUARED is False:
                pytest.skip()
            # create input
            autograd_res = self.test_problem.sgs_autograd()
            bpexts_res = self.test_problem.sgs_bpexts()
            layer = self.test_problem.model
            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for g1, g2, p in zip(autograd_res, bpexts_res, layer.parameters()):
                assert g1.size() == g2.size() == p.size()
                self._residuum_report(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_diag_ggn(self):
            """Test the diagonal of the GGN."""
            if self.TEST_DIAG_GGN is False:
                pytest.skip()
            layer = self.test_problem.model

            autograd_res = self.test_problem.diag_ggn_autograd()
            bpexts_res = self.test_problem.diag_ggn_bpexts()

            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for ggn1, ggn2, p in zip(autograd_res, bpexts_res,
                                     layer.parameters()):
                assert ggn1.size() == ggn2.size() == p.size()
                self._residuum_report(ggn1, ggn2)
                assert torch.allclose(
                    ggn1, ggn2, atol=self.ATOL, rtol=self.RTOL)

        def _residuum_report(self, x, y):
            """Report values with mismatch in allclose check."""
            x_numpy = x.data.cpu().numpy().flatten()
            y_numpy = y.data.cpu().numpy().flatten()
            close = numpy.isclose(
                x_numpy, y_numpy, atol=self.ATOL, rtol=self.RTOL)
            where_not_close = numpy.argwhere(numpy.logical_not(close))
            for idx in where_not_close:
                print('{} versus {}, {}, {}'.format(x_numpy[idx], y_numpy[idx], x_numpy[idx] / y_numpy[idx], y_numpy[idx] / x_numpy[idx]))
                break

    class GradientTest1(GradientTest0):
        # loss Hessian is not PD
        TEST_DIAG_GGN = False

        test_problem = make_test_problem(seed, layer_fn, input_size, loss1, device)

    class GradientTest2(GradientTest0):
        # loss Hessian is not PD
        TEST_DIAG_GGN = False

        test_problem = make_test_problem(seed, layer_fn, input_size, loss2, device)

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

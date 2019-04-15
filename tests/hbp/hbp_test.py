"""Testing class for comparison of torch and HBP layers."""

import torch
import numpy
import unittest
from bpexts.utils import set_seeds
from bpexts.hessian.exact import exact_hessian
from bpexts.hbp.loss import batch_summed_hessian


def hbp_test(torch_fn,
             hbp_fn,
             input_size,
             device,
             seed=0,
             atol=1e-5,
             rtol=1e-8,
             num_hvp=10):
    """Create unittest for checking behavior of torch layer and correct HBP.

    For HBP to be exact, the batch size has to be 1!

    Checks for the following aspects:

    - same initialization of parameters
    - same result in a forward and backward pass (using different dummy losses)
    - same Hessian-vector products for the input Hessian
    - same Hessian-vector products for the block-wise parameter Hessians

    Parameters:
    -----------
    torch_fn : function
        Function creating the torch layer. Parameters have to be initialized
        in the same way as by ``hbp_fn``
    hbp_fn : function
        Function creating the HBP layer.
    input_size : tuple(int)
        Dimension of the input
    device : torch.device
        Device to run the tests on
    atol : float
        Absolute tolerance for elementwise comparison of HVP results
    rtol : float
        Relative tolerance for elementwise comparison of HVP results
    num_hvp : int
        Number of tested Hessian-vector with random vectors

    Returns:
    --------
    unittest.TestCase
        test case for comparing the functionalities of torch and hbp layers
    """
    if not input_size[0] == 1:
        raise ValueError('HBP is only exact for a single sample.')

    class HBPTest0(unittest.TestCase):
        SEED = seed
        DEVICE = device
        INPUT_SIZE = input_size
        ATOL = atol
        RTOL = rtol
        NUM_HVP = num_hvp

        def torch_fn(self):
            """Create the torch layer."""
            return torch_fn()

        def hbp_fn(self):
            """Create the HBP layer."""
            return hbp_fn()

        def _loss_fn(self, x):
            """Dummy loss function: Normalized sum of squared elements.

            The loss Hessian of this function is constant and diagonal.
            """
            return (x**2).contiguous().view(-1).sum() / x.numel()

        def test_same_parameters(self):
            """Check if parameters are initialized the same way."""
            torch_layer, hbp_layer = self._create_layers()
            assert len(list(torch_layer.parameters())) == len(
                list(hbp_layer.parameters()))
            for torch_param, hbp_param in zip(torch_layer.parameters(),
                                              hbp_layer.parameters()):
                assert torch.allclose(torch_param, hbp_param)

        def test_forward_pass(self):
            """Check for same forward pass."""
            # create input
            torch_in, hbp_in = self._create_input(), self._create_input()
            torch_layer, hbp_layer = self._create_layers()
            torch_out, hbp_out = torch_layer(torch_in), hbp_layer(hbp_in)
            assert torch.allclose(torch_out, hbp_out)

        def test_backward_pass(self):
            """Check for same backward pass."""
            torch_in, hbp_in = self._create_input(), self._create_input()
            torch_in.requires_grad = True
            hbp_in.requires_grad = True
            torch_layer, hbp_layer = self._create_layers()
            torch_out, hbp_out = torch_layer(torch_in), hbp_layer(hbp_in)
            torch_loss, hbp_loss = self._loss_fn(torch_out), self._loss_fn(
                hbp_out)
            torch_loss.backward()
            hbp_loss.backward()
            assert torch.allclose(torch_in.grad, hbp_in.grad)
            for torch_param, hbp_param in zip(torch_layer.parameters(),
                                              hbp_layer.parameters()):
                assert torch.allclose(torch_param.grad, hbp_param.grad)

        def test_parameter_hvp(self):
            """Compare the parameter HVPs."""
            hbp_layer = self._hbp_after_hessian_backward()
            torch_parameter_hessians = list(self._torch_parameter_hvp())
            for _ in range(self.NUM_HVP):
                for idx, p in enumerate(hbp_layer.parameters()):
                    v = torch.randn(p.numel()).to(self.DEVICE)
                    torch_hvp = torch_parameter_hessians[idx].matmul
                    hbp_result = p.hvp(v)
                    torch_result = torch_hvp(v)
                    self._residuum_report(hbp_result, torch_result)
                    assert torch.allclose(
                        hbp_result,
                        torch_result,
                        atol=self.ATOL,
                        rtol=self.RTOL)

        def test_input_hvp(self):
            """Compare HVP with respect to the input."""
            torch_hvp = self._torch_input_hvp()
            hbp_hvp = self._hbp_input_hvp()
            for _ in range(self.NUM_HVP):
                input_numel = int(numpy.prod(self.INPUT_SIZE))
                v = torch.randn(input_numel).to(self.DEVICE)
                torch_result = torch_hvp(v)
                hbp_result = hbp_hvp(v)
                self._residuum_report(hbp_result, torch_result)
                assert torch.allclose(
                    torch_result, hbp_result, atol=self.ATOL, rtol=self.RTOL)

        def _residuum_report(self, x, y):
            """Report values with mismatch in allclose check."""
            x_numpy = x.data.cpu().numpy()
            y_numpy = y.data.cpu().numpy()
            close = numpy.isclose(
                x_numpy, y_numpy, atol=self.ATOL, rtol=self.RTOL)
            where_not_close = numpy.argwhere(numpy.logical_not(close))
            for idx in where_not_close:
                print('{} versus {}'.format(x_numpy[idx], y_numpy[idx]))

        def _torch_input_hvp(self):
            """Create Hessian-vector product routine for torch layer."""
            layer = self._create_torch_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            hessian_x = exact_hessian(loss, [x]).detach().to(self.DEVICE)
            return hessian_x.matmul

        def _torch_parameter_hvp(self):
            """Yield block-wise HVP with the parameter Hessians"""
            layer = self._create_torch_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            for p in layer.parameters():
                yield exact_hessian(loss, [p]).detach().to(self.DEVICE)

        def _hbp_input_hvp(self):
            """Create Hessian-vector product routine for HBP layer."""
            layer = self._create_hbp_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            # required for nonlinear layers (need to save backprop quantities)
            loss_hessian = batch_summed_hessian(loss, out).detach()
            loss.backward()
            hessian_x = layer.backward_hessian(loss_hessian)
            return hessian_x.matmul

        def _hbp_after_hessian_backward(self):
            """Return the HBP layer after performing HBP."""
            layer = self._create_hbp_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            # required for nonlinear layers (need to save backprop quantities)
            loss_hessian = batch_summed_hessian(loss, out).detach()
            loss.backward()
            layer.backward_hessian(loss_hessian)
            return layer

        def _create_input(self):
            """Return same random input (reset seed before)."""
            set_seeds(self.SEED)
            return torch.randn(*self.INPUT_SIZE).to(self.DEVICE)

        def _create_layers(self):
            """Create both torch and the HBP layer."""
            return self._create_torch_layer(), self._create_hbp_layer()

        def _create_torch_layer(self):
            """Create the torch layer."""
            return self.torch_fn().to(self.DEVICE)

        def _create_hbp_layer(self):
            """Create the HBP layer."""
            return self.hbp_fn().to(self.DEVICE)

    class HBPTest1(HBPTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Normalized cubed sum.

            The loss Hessian of this function is constant and non-diagonal.
            """
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += (x[b, :].view(-1).sum())**2 / x.numel()
            return loss

    class HBPTest2(HBPTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Sum of log10 of shifted normalized abs.

            The loss Hessian of this function is non-constant and non-diagonal.
            """
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += ((torch.log10(torch.abs(x[b, :]) + 0.1) /
                          x.numel()).view(-1).sum())**2
            return loss

    return HBPTest0, HBPTest1, HBPTest2


def set_up_hbp_tests(torch_fn,
                     hbp_fn,
                     layer_name,
                     input_size,
                     atol=1e-8,
                     rtol=1e-5,
                     num_hvp=10):
    """Yield the names and classes for the unittests."""
    # create CPU tests
    for idx, cpu_test in enumerate(
            hbp_test(
                torch_fn,
                hbp_fn,
                input_size,
                device=torch.device('cpu'),
                atol=atol,
                rtol=rtol,
                num_hvp=num_hvp)):
        print(cpu_test.DEVICE)
        yield '{}CPUTest{}'.format(layer_name, idx), cpu_test

    # create GPU tests if available
    if torch.cuda.is_available():
        for idx, gpu_test in enumerate(
                hbp_test(
                    torch_fn,
                    hbp_fn,
                    input_size,
                    device=torch.device('cuda:0'),
                    atol=atol,
                    rtol=rtol,
                    num_hvp=num_hvp)):
            print(gpu_test.DEVICE)
            yield '{}GPUTest{}'.format(layer_name, idx), gpu_test

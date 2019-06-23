"""Testing class for comparison of torch and CVP layers."""

import torch
import numpy
import unittest
from bpexts.utils import set_seeds
from bpexts.hessian.exact import exact_hessian
from bpexts.cvp.sequential import CVPSequential
from bpexts.cvp.flatten import Flatten
from bpexts.hessian.free import transposed_jacobian_vector_product, jacobian_vector_product


def cvp_test(torch_fn,
             cvp_fn,
             input_size,
             device,
             seed=0,
             atol=1e-5,
             rtol=1e-8,
             num_hvp=10):
    """Create unittest for checking behavior of torch layer and correct CVP.

    Checks for the following aspects:

    - same initialization of parameters
    - same result in a forward and backward pass (using different dummy losses)
    - same Hessian-vector products for the input Hessian
    - same Hessian-vector products for the block-wise parameter Hessians

    Parameters:
    -----------
    torch_fn : function
        Function creating the torch layer. Parameters have to be initialized
        in the same way as by ``cvp_fn``
    cvp_fn : function
        Function creating the CVP layer.
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
        test case for comparing the functionalities of torch and CVP layers
    """

    class CVPTest0(unittest.TestCase):
        SEED = seed
        DEVICE = device
        INPUT_SIZE = input_size
        ATOL = atol
        RTOL = rtol
        NUM_HVP = num_hvp

        def torch_fn(self):
            """Create the torch layer."""
            return torch_fn()

        def cvp_fn(self):
            """Create the CVP layer."""
            return cvp_fn()

        def _loss_fn(self, x):
            """Dummy loss function: Normalized sum of squared elements.

            The loss Hessian of this function is constant and diagonal.
            """
            return (x**2).contiguous().view(-1).sum() / x.numel()

        def test_same_parameters(self):
            """Check if parameters are initialized the same way."""
            torch_layer, cvp_layer = self._create_layers()
            assert len(list(torch_layer.parameters())) == len(
                list(cvp_layer.parameters()))
            for torch_param, cvp_param in zip(torch_layer.parameters(),
                                              cvp_layer.parameters()):
                assert torch.allclose(torch_param, cvp_param)

        def test_forward_pass(self):
            """Check for same forward pass."""
            # create input
            torch_in, cvp_in = self._create_input(), self._create_input()
            torch_layer, cvp_layer = self._create_layers()
            torch_out, cvp_out = torch_layer(torch_in), cvp_layer(cvp_in)
            assert torch.allclose(torch_out, cvp_out)

        def test_backward_pass(self):
            """Check for same backward pass."""
            torch_in, cvp_in = self._create_input(), self._create_input()
            torch_in.requires_grad = True
            cvp_in.requires_grad = True
            torch_layer, cvp_layer = self._create_layers()
            torch_out, cvp_out = torch_layer(torch_in), cvp_layer(cvp_in)
            torch_loss, cvp_loss = self._loss_fn(torch_out), self._loss_fn(
                cvp_out)
            torch_loss.backward()
            cvp_loss.backward()
            assert torch.allclose(torch_in.grad, cvp_in.grad)
            for torch_param, cvp_param in zip(torch_layer.parameters(),
                                              cvp_layer.parameters()):
                assert torch.allclose(torch_param.grad, cvp_param.grad)

        def test_parameter_hvp(self):
            """Compare the parameter HVPs."""
            cvp_layer = self._cvp_after_hessian_backward()
            torch_parameter_hessians = list(self._torch_parameter_hvp())
            for _ in range(self.NUM_HVP):
                for idx, p in enumerate(cvp_layer.parameters()):
                    v = torch.randn(
                        p.numel(), requires_grad=False).to(self.DEVICE)
                    torch_hvp = torch_parameter_hessians[idx].matmul
                    cvp_result = p.hvp(v)
                    assert cvp_result.requires_grad == False
                    torch_result = torch_hvp(v)
                    assert torch_result.requires_grad == False
                    self._residuum_report(cvp_result, torch_result)
                    assert torch.allclose(
                        cvp_result,
                        torch_result,
                        atol=self.ATOL,
                        rtol=self.RTOL)

        def test_input_hvp(self):
            """Compare HVP with respect to the input."""
            torch_hvp = self._torch_input_hvp()
            cvp_hvp = self._cvp_input_hvp()
            for _ in range(self.NUM_HVP):
                input_numel = int(numpy.prod(self.INPUT_SIZE))
                v = torch.randn(
                    input_numel, requires_grad=False).to(self.DEVICE)
                torch_result = torch_hvp(v)
                assert torch_result.requires_grad == False
                cvp_result = cvp_hvp(v)
                assert cvp_result.requires_grad == False
                self._residuum_report(cvp_result, torch_result)
                assert torch.allclose(
                    torch_result, cvp_result, atol=self.ATOL, rtol=self.RTOL)

        def test_input_jacobian(self):
            """Test multiplication by the input Jacobian.

            Compare with result of R-operator.
            """
            # create input
            cvp_in = self._create_input()
            cvp_in.requires_grad = True
            cvp_layer = self._create_cvp_layer()
            # skip for Sequential:
            if isinstance(cvp_layer, (Flatten, CVPSequential)):
                return
            cvp_out = cvp_layer(cvp_in)
            for _ in range(self.NUM_HVP):
                v = torch.randn(
                    cvp_in.numel(), requires_grad=False).to(self.DEVICE)
                Jv = cvp_layer._input_jacobian(v)
                # compute via R-operator
                result, = jacobian_vector_product(cvp_out, cvp_in,
                                                  v.view(cvp_in.size()))
                assert torch.allclose(
                    Jv, result.view(-1), atol=self.ATOL, rtol=self.RTOL)

        def test_input_jacobian_transpose(self):
            """Test multiplication by the transposed input Jacobian.

            Compare with result of L-operator.
            """
            # create input
            cvp_in = self._create_input()
            cvp_in.requires_grad = True
            cvp_layer = self._create_cvp_layer()
            # skip for Sequential:
            if isinstance(cvp_layer, (Flatten, CVPSequential)):
                return
            cvp_out = cvp_layer(cvp_in)
            for _ in range(self.NUM_HVP):
                v = torch.randn(
                    cvp_out.numel(), requires_grad=False).to(self.DEVICE)
                JTv = cvp_layer._input_jacobian_transpose(v)
                # compute via L-operator
                result, = transposed_jacobian_vector_product(
                    cvp_out, cvp_in, v.view(cvp_out.size()))
                assert torch.allclose(
                    JTv, result.view(-1), atol=self.ATOL, rtol=self.RTOL)

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

        def _cvp_input_hvp(self):
            """Create Hessian-vector product routine for CVP layer."""
            layer = self._create_cvp_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            # required for nonlinear layers (need to save backprop quantities)
            loss_hessian_vp = exact_hessian(loss, [out]).detach().to(
                self.DEVICE).matmul
            loss.backward()
            hessian_x = layer.backward_hessian(
                loss_hessian_vp, compute_input_hessian=True)
            return hessian_x

        def _cvp_after_hessian_backward(self):
            """Return the CVP layer after performing CVP."""
            layer = self._create_cvp_layer()
            x = self._create_input()
            x.requires_grad = True
            out = layer(x)
            loss = self._loss_fn(out)
            # required for nonlinear layers (need to save backprop quantities)
            loss_hessian_vp = exact_hessian(loss, [out]).detach().to(
                self.DEVICE).matmul
            loss.backward()
            layer.backward_hessian(loss_hessian_vp)
            return layer

        def _create_input(self):
            """Return same random input (reset seed before)."""
            set_seeds(self.SEED)
            return torch.randn(*self.INPUT_SIZE).to(self.DEVICE)

        def _create_layers(self):
            """Create both torch and the CVP layer."""
            return self._create_torch_layer(), self._create_cvp_layer()

        def _create_torch_layer(self):
            """Create the torch layer."""
            return self.torch_fn().to(self.DEVICE)

        def _create_cvp_layer(self):
            """Create the CVPlayer."""
            return self.cvp_fn().to(self.DEVICE)

    class CVPTest1(CVPTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Normalized cubed sum.

            The loss Hessian of this function is constant and non-diagonal.
            """
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += (x[b, :].view(-1).sum())**2 / x.numel()
            return loss

    class CVPTest2(CVPTest0):
        def _loss_fn(self, x):
            """Dummy loss function: Sum of log10 of shifted normalized abs.

            The loss Hessian of this function is non-constant and non-diagonal.
            """
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += ((torch.log10(torch.abs(x[b, :]) + 0.1) /
                          x.numel()).view(-1).sum())**2
            return loss

    return CVPTest0, CVPTest1, CVPTest2


def set_up_cvp_tests(torch_fn,
                     cvp_fn,
                     layer_name,
                     input_size,
                     atol=1e-8,
                     rtol=1e-5,
                     num_hvp=10):
    """Yield the names and classes for the unittests."""
    # create CPU tests
    for idx, cpu_test in enumerate(
            cvp_test(
                torch_fn,
                cvp_fn,
                input_size,
                device=torch.device('cpu'),
                atol=atol,
                rtol=rtol,
                num_hvp=num_hvp)):
        yield '{}CPUTest{}'.format(layer_name, idx), cpu_test

    # create GPU tests if available
    if torch.cuda.is_available():
        for idx, gpu_test in enumerate(
                cvp_test(
                    torch_fn,
                    cvp_fn,
                    input_size,
                    device=torch.device('cuda:0'),
                    atol=atol,
                    rtol=rtol,
                    num_hvp=num_hvp)):
            yield '{}GPUTest{}'.format(layer_name, idx), gpu_test

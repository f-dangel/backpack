"""Testing class for comparison of torch and CVP loss functions."""

import torch
import numpy
import unittest
from bpexts.utils import set_seeds
from bpexts.hessian.exact import exact_hessian


def cvp_test(torch_fn,
             cvp_fn,
             input_size,
             device,
             seed=0,
             atol=1e-5,
             rtol=1e-8,
             num_hvp=10):
    """Create unittest for checking behavior of torch loss and correct CVP.

    Checks for the following aspects:

    - same result in a forward and backward pass (using different dummy losses)
    - same Hessian-vector products for the input Hessian

    Parameters:
    -----------
    torch_fn : function
        Function creating the torch loss. Parameters have to be initialized
        in the same way as by ``cvp_fn``
    cvp_fn : function
        Function creating the CVP loss.
    input_size : tuple(int)
        Dimension of the input, first dimension is batch, second is classes
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
        test case for comparing the functionalities of torch and CVP losses
    """
    assert len(input_size) == 2

    class CVPLossTest(unittest.TestCase):
        SEED = seed
        DEVICE = device
        INPUT_SIZE = input_size
        BATCH, NUM_CLASSES = input_size
        ATOL = atol
        RTOL = rtol
        NUM_HVP = num_hvp

        def test_forward_pass(self):
            """Check for same forward pass."""
            # create input
            torch_in, torch_labels = self._create_input()
            cvp_in, cvp_labels = self._create_input()
            torch_layer, cvp_layer = self._create_layers()
            torch_out = torch_layer(torch_in, torch_labels)
            cvp_out = cvp_layer(cvp_in, cvp_labels)
            assert torch.allclose(torch_out, cvp_out)

        def test_backward_pass(self):
            """Check for same backward pass."""
            torch_in, torch_labels = self._create_input()
            cvp_in, cvp_labels = self._create_input()
            torch_in.requires_grad = True
            cvp_in.requires_grad = True
            torch_layer, cvp_layer = self._create_layers()
            torch_out = torch_layer(torch_in, torch_labels)
            cvp_out = cvp_layer(cvp_in, cvp_labels)
            torch_out.backward()
            cvp_out.backward()
            assert torch.allclose(torch_in.grad, cvp_in.grad)
            for torch_param, cvp_param in zip(torch_layer.parameters(),
                                              cvp_layer.parameters()):
                assert torch.allclose(torch_param.grad, cvp_param.grad)

        def test_input_hvp(self):
            """Compare HVP with respect to the input."""
            torch_hvp = self._torch_input_hvp()
            cvp_hvp = self._cvp_input_hvp()
            for _ in range(self.NUM_HVP):
                input_numel = int(numpy.prod(self.INPUT_SIZE))
                v = torch.randn(
                    input_numel, requires_grad=False).to(self.DEVICE)
                torch_result = torch_hvp(v)
                cvp_result = cvp_hvp(v)
                print(self._residuum(cvp_result, torch_result))
                assert torch.allclose(
                    torch_result, cvp_result, atol=self.ATOL, rtol=self.RTOL)

        @staticmethod
        def _residuum(x, y):
            """Maximum of absolute value difference."""
            return torch.max(torch.abs(x - y))

        def _torch_input_hvp(self):
            """Create Hessian-vector product routine for torch layer."""
            layer = self._create_torch_layer()
            x, y = self._create_input()
            x.requires_grad = True
            out = layer(x, y)
            hessian_x = exact_hessian(out, [x]).detach().to(self.DEVICE)
            return hessian_x.matmul

        def _cvp_input_hvp(self):
            """Create Hessian-vector product routine for CVP layer."""
            layer = self._create_cvp_layer()
            x, y = self._create_input()
            x.requires_grad = True
            out = layer(x, y)
            loss_hessian_vp = None
            out.backward()
            hessian_x = layer.backward_hessian(loss_hessian_vp)
            return hessian_x

        def _create_input(self):
            """Return same random input (reset seed before) and labels."""
            set_seeds(self.SEED)
            x = torch.randn(*self.INPUT_SIZE).to(self.DEVICE)
            y = torch.randint(
                size=(self.BATCH, ), low=0,
                high=self.NUM_CLASSES).to(self.DEVICE)
            return x, y

        def _create_layers(self):
            """Create both torch and the CVP layer."""
            return self._create_torch_layer(), self._create_cvp_layer()

        def _create_torch_layer(self):
            """Create the torch layer."""
            return torch_fn().to(self.DEVICE)

        def _create_cvp_layer(self):
            """Create the CVPlayer."""
            return cvp_fn().to(self.DEVICE)

    return CVPLossTest


def set_up_cvp_loss_tests(torch_fn,
                          cvp_fn,
                          layer_name,
                          input_size,
                          atol=1e-8,
                          rtol=1e-5,
                          num_hvp=10):
    """Yield the names and classes for the unittests."""
    # create CPU tests
    yield '{}CPUTest'.format(layer_name), cvp_test(
        torch_fn,
        cvp_fn,
        input_size,
        device=torch.device('cpu'),
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp)

    # create GPU tests if available
    if torch.cuda.is_available():
        yield '{}GPUTest'.format(layer_name), cvp_test(
            torch_fn,
            cvp_fn,
            input_size,
            device=torch.device('cuda:0'),
            atol=atol,
            rtol=rtol,
            num_hvp=num_hvp)

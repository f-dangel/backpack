"""Testing class for comparison of first-order information and brute-force
 auto-differentiation."""

import torch
import pytest
import numpy
import unittest
from bpexts.utils import set_seeds
import bpexts.gradient.config as config
import bpexts.hessian.free as hessian_free
import bpexts.hessian.exact as exact
import bpexts.utils as utils


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

        def _loss_fn(self, x):
            """Dummy loss function: Normalized sum of squared elements."""
            return (x**2).contiguous().view(x.size(0), -1).mean(0).sum()

        def test_batch_gradients(self):
            """Check for same batch gradients."""
            # create input
            autograd_res = self._compute_batch_gradients_autograd()
            bpexts_res = self._compute_batch_gradients_bpexts()
            layer = self._create_layer()
            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for g1, g2, p in zip(autograd_res, bpexts_res, layer.parameters()):
                assert tuple(g1.size()) == tuple(
                    g2.size()) == (self.INPUT_SIZE[0], ) + tuple(p.size())
                self._residuum_report(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def _compute_batch_gradients_autograd(self):
            """Batch gradients via torch.autograd."""
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
            """Batch gradients via bpexts."""
            layer = self._create_layer()
            inputs = self._create_input()
            loss = self._loss_fn(layer(inputs))
            with config.bpexts(config.BATCH_GRAD):
                loss.backward()
                batch_grads = [p.grad_batch for p in layer.parameters()]
                for p in layer.parameters():
                    del p.grad
                    del p.grad_batch
            return batch_grads

        def test_batch_gradients_sum_to_grad(self):
            """Check if the batch sum of gradients yields the gradient."""
            layer = self._create_layer()
            autograd_gradients = self._compute_gradients_autograd()
            bpexts_batch_gradients = self._compute_batch_gradients_bpexts()
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

        def _compute_gradients_autograd(self):
            """Gradients via torch.autograd."""
            layer = self._create_layer()
            inputs = self._create_input()
            loss = self._loss_fn(layer(inputs))
            gradients = torch.autograd.grad(loss, layer.parameters())
            return list(gradients)

        def test_sgs(self):
            """Check for same sum of gradients squared."""
            if self.TEST_SUM_GRAD_SQUARED is False:
                pytest.skip()
            # create input
            autograd_res = self._compute_sgs_autograd()
            bpexts_res = self._compute_sgs_bpexts()
            layer = self._create_layer()
            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for g1, g2, p in zip(autograd_res, bpexts_res, layer.parameters()):
                assert g1.size() == g2.size() == p.size()
                self._residuum_report(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def _compute_sgs_autograd(self):
            """Sum of squared gradients via torch.autograd."""
            batch_grad = self._compute_batch_gradients_autograd()
            sgs = [(g**2).sum(0) for g in batch_grad]
            return sgs

        def _compute_sgs_bpexts(self):
            """Sum of squared gradients via bpexts."""
            layer = self._create_layer()
            inputs = self._create_input()
            loss = self._loss_fn(layer(inputs))
            with config.bpexts(config.SUM_GRAD_SQUARED):
                loss.backward()
                sgs = [p.sum_grad_squared for p in layer.parameters()]

                for p in layer.parameters():
                    del p.grad
                    del p.sum_grad_squared
            return sgs

        def test_diag_ggn(self):
            """Test the diagonal of the GGN."""
            if self.TEST_DIAG_GGN is False:
                pytest.skip()
            layer = self._create_layer()
            autograd_res = self._compute_diag_ggn_autograd()
            bpexts_res = self._compute_diag_ggn_bpexts()
            assert len(autograd_res) == len(bpexts_res) == len(
                list(layer.parameters()))
            for ggn1, ggn2, p in zip(autograd_res, bpexts_res,
                                     layer.parameters()):
                assert ggn1.size() == ggn2.size() == p.size()
                self._residuum_report(ggn1, ggn2)
                assert torch.allclose(
                    ggn1, ggn2, atol=self.ATOL, rtol=self.RTOL)

        def _compute_diag_ggn_autograd(self):
            """Diagonal of the GGN matrix via autodiff."""
            layer = self._create_layer()
            inputs = self._create_input()
            outputs = layer(inputs)
            loss = self._loss_fn(outputs)

            tot_params = sum([p.numel() for p in layer.parameters()])
            i = 0
            diag_ggns = []

            for p in list(layer.parameters()):
                diag_ggn_p = torch.zeros_like(p).view(-1)
                # extract i.th column of the GGN, store diagonal entry
                for i_p in range(p.numel()):
                    v = torch.zeros(tot_params).to(self.DEVICE)
                    v[i] = 1.

                    vs = hessian_free.vector_to_parameter_list(
                        v, list(layer.parameters()))

                    # GGN-vector product
                    GGN_v = hessian_free.ggn_vector_product(
                        loss, outputs, layer, vs)
                    GGN_v = torch.cat([g.detach().view(-1) for g in GGN_v])

                    diag_ggn_p[i_p] = GGN_v[i]
                    i += 1
                # reshape into parameter dimension and append
                diag_ggns.append(diag_ggn_p.view(p.size()))
            return diag_ggns

        def _compute_diag_ggn_bpexts(self):
            """Diagonal of the GGN matrix via bpexts.

            Only works for loss functions with PD Hessian w.r.t.
            the model output.
            """
            layer = self._create_layer()
            inputs = self._create_input()
            outputs = layer(inputs)
            loss = self._loss_fn(outputs)
            with config.bpexts(config.DIAG_GGN):
                # need to set manually
                sqrt_loss_hessian = self._compute_loss_hessian_cholesky()
                config.CTX._backpropagated_sqrt_ggn = sqrt_loss_hessian
                # do the backward pass
                loss.backward()
                diag_ggns = [p.diag_ggn for p in layer.parameters()]
            return diag_ggns

        def _compute_loss_hessian_autograd(self):
            """Compute the Hessian of the individual loss w.r.t. the output.

            Return a tensor storing the loss Hessians for each sample along
            the first dimension.
            """
            layer = self._create_layer()
            inputs = self._create_input()
            loss_hessians = []
            for b in range(inputs.size(0)):
                input = inputs[b, :].unsqueeze(0)
                output = layer(input)
                loss = self._loss_fn(output) / inputs.size(0)

                h = exact.exact_hessian(loss, [output]).detach()
                loss_hessians.append(h.detach())

            loss_hessians = torch.stack(loss_hessians)
            assert tuple(loss_hessians.size()) == (inputs.size(0),
                                                   output.numel(),
                                                   output.numel())
            return loss_hessians

        def _compute_loss_hessian_cholesky(self):
            """Return batch-wise matrix sqrt of the loss Hessian."""
            loss_hessian = self._compute_loss_hessian_autograd()
            return torch.cholesky(loss_hessian)

        def _create_layer(self):
            """Create layer and load to device."""
            return layer_fn().to(self.DEVICE)

        def _create_input(self):
            """Return same random input (reset seed before)."""
            set_seeds(self.SEED)
            return torch.randn(self.INPUT_SIZE).to(self.DEVICE)

        def _residuum_report(self, x, y):
            """Report values with mismatch in allclose check."""
            x_numpy = x.data.cpu().numpy().flatten()
            y_numpy = y.data.cpu().numpy().flatten()
            close = numpy.isclose(
                x_numpy, y_numpy, atol=self.ATOL, rtol=self.RTOL)
            where_not_close = numpy.argwhere(numpy.logical_not(close))
            for idx in where_not_close:
                print('{} versus {}'.format(x_numpy[idx], y_numpy[idx]))

    class GradientTest1(GradientTest0):
        # loss Hessian is not PD
        TEST_DIAG_GGN = False

        def _loss_fn(self, x):
            """Dummy loss function: Normalized squared sum."""
            loss = torch.zeros(1).to(self.DEVICE)
            for b in range(x.size(0)):
                loss += (x[b, :].view(-1).sum())**2 / x.size(0)
            return loss

    class GradientTest2(GradientTest0):
        # loss Hessian is not PD
        TEST_DIAG_GGN = False

        def _loss_fn(self, x):
            """Dummy loss function: Sum of log10 of shifted normalized abs."""
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

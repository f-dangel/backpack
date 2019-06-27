import torch
import pytest
import numpy as np
import unittest


def unittest_for(test_problem, atol=1e-5, rtol=1e-8):
    """Create unittest for checking the test_problem.

    Checks for the following aspects:
    - batchwise gradients
    - sum of gradients squared

    Parameters:
    -----------
    test_problem : bpext.tests.gradient.test_problem.TestProblem
    atol : float
        Absolute tolerance for elementwise comparison
    rtol : float
        Relative tolerance for elementwise comparison

    Returns:
    --------
    unittest.TestCase
        test case for comparing the first-order functionalities with autograd
    """

    class GradientTest(unittest.TestCase):
        ATOL = atol
        RTOL = rtol
        TEST_DIAG_GGN = False

        problem = test_problem

        def test_batch_gradients(self):
            autograd_res = self.problem.batch_gradients_autograd()
            bpexts_res = self.problem.batch_gradients_bpexts()
            model = self.problem.model

            assert len(autograd_res) == len(bpexts_res) == len(
                list(model.parameters()))
            for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
                assert tuple(g1.size()) == tuple(
                    g2.size()) == (self.problem.N, ) + tuple(p.size())
                self.report_nonclose_values(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_batch_gradients_sum_to_grad(self):
            model = self.problem.model
            autograd_gradients = self.problem.gradient_autograd()
            bpexts_batch_gradients = self.problem.batch_gradients_bpexts()

            assert len(autograd_gradients) == len(
                bpexts_batch_gradients) == len(list(model.parameters()))
            for g, batch_g, p in zip(
                    autograd_gradients, bpexts_batch_gradients,
                    model.parameters()):
                bpexts_g = batch_g.sum(0)
                assert g.size() == bpexts_g.size() == p.size()
                self.report_nonclose_values(g, bpexts_g)
                assert torch.allclose(
                    g, bpexts_g, atol=self.ATOL, rtol=self.RTOL)

        def test_sgs(self):
            autograd_res = self.problem.sgs_autograd()
            bpexts_res = self.problem.sgs_bpexts()

            model = self.problem.model
            assert len(autograd_res) == len(bpexts_res) == len(list(model.parameters()))

            for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
                assert g1.size() == g2.size() == p.size()
                self.report_nonclose_values(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        @pytest.mark.skip(reason="GGN not ready")
        def test_diag_ggn(self):
            model = self.problem.model

            autograd_res = self.problem.diag_ggn_autograd()
            bpexts_res = self.problem.diag_ggn_bpexts()

            assert len(autograd_res) == len(bpexts_res) == len(list(model.parameters()))
            for ggn1, ggn2, p in zip(autograd_res, bpexts_res,
                                     model.parameters()):
                assert ggn1.size() == ggn2.size() == p.size()
                self.report_nonclose_values(ggn1, ggn2)
                assert torch.allclose(
                    ggn1, ggn2, atol=self.ATOL, rtol=self.RTOL)

        def report_nonclose_values(self, x, y):
            x_numpy = x.data.cpu().numpy().flatten()
            y_numpy = y.data.cpu().numpy().flatten()

            close = np.isclose(
                x_numpy, y_numpy, atol=self.ATOL, rtol=self.RTOL)
            where_not_close = np.argwhere(np.logical_not(close))
            for idx in where_not_close:
                print('{} versus {}'.format(x_numpy[idx], y_numpy[idx]))

    return GradientTest

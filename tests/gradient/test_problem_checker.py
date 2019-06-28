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

            self.check_sizes(autograd_res, bpexts_res)

            for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
                self.report_nonclose_values(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_batch_gradients_sum_to_grad(self):
            model = self.problem.model
            autograd_res = self.problem.gradient_autograd()
            bpexts_batch_res = self.problem.batch_gradients_bpexts()
            bpexts_res = list([g.sum(0) for g in bpexts_batch_res])

            self.check_sizes(autograd_res, bpexts_res, list(model.parameters()))
            for g1, g2, p in zip(
                    autograd_res, bpexts_res,
                    model.parameters()):
                self.report_nonclose_values(g1, g2)
                assert torch.allclose(
                    g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_sgs(self):
            autograd_res = self.problem.sgs_autograd()
            bpexts_res = self.problem.sgs_bpexts()

            model = self.problem.model

            self.check_sizes(autograd_res, bpexts_res, list(model.parameters()))

            for g1, g2, p in zip(autograd_res, bpexts_res, model.parameters()):
                self.report_nonclose_values(g1, g2)
                assert torch.allclose(g1, g2, atol=self.ATOL, rtol=self.RTOL)

        def test_diag_ggn(self):
            model = self.problem.model

            autograd_res = self.problem.diag_ggn_autograd()
            bpexts_res = self.problem.diag_ggn_bpexts()

            self.check_sizes(autograd_res, bpexts_res, list(model.parameters()))

            for ggn1, ggn2, p in zip(autograd_res, bpexts_res,
                                     model.parameters()):
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
                x, y = x_numpy[idx], y_numpy[idx]
                print('{} versus {}. Ratio of {}'.format(x, y, y / x))

        def check_sizes(self, *plists):
            for i in range(len(plists) - 1):
                assert len(plists[i]) == len(plists[i + 1])

            for params in zip(*plists):
                for i in range(len(params) - 1):
                    assert params[i].size() == params[i + 1].size()

    return GradientTest

"""Test of Kronecker utilities."""

import random
import unittest

import torch

import scipy.linalg
from backpack.extensions.secondorder import utils as bp_utils


class KroneckerUtilsTest(unittest.TestCase):
    RUNS = 100

    # Precision of results
    ATOL = 1e-6
    RTOL = 1e-5

    # Restriction of dimension and number of factors
    MIN_DIM = 1
    MAX_DIM = 5
    MIN_FACS = 1
    MAX_FACS = 3

    # Number of columns for KFAC-matrix products
    KFACMP_COLS = 7

    # HELPERS
    ##########################################################################
    def allclose(self, tensor1, tensor2):
        return torch.allclose(tensor1, tensor2, rtol=self.RTOL, atol=self.ATOL)

    def make_random_kfacs(self, num_facs=None, quadratic=False):
        def random_kfac():
            def random_dim():
                return random.randint(self.MIN_DIM, self.MAX_DIM)

            def random_shape():
                if quadratic:
                    return 2 * [random_dim()]
                else:
                    return [random_dim(), random_dim()]

            shape = random_shape()
            return torch.rand(shape)

        def random_num_facs():
            return random.randint(self.MIN_FACS, self.MAX_FACS)

        num_facs = num_facs if num_facs is not None else random_num_facs()
        return [random_kfac() for _ in range(num_facs)]

    # SCIPY implementations
    ##########################################################################

    def scipy_two_kfacs_to_mat(self, A, B):
        return torch.from_numpy(scipy.linalg.kron(A.numpy(), B.numpy()))

    def scipy_kfacs_to_mat(self, factors):
        mat = None
        for factor in factors:
            if mat is None:
                assert bp_utils.is_matrix(factor)
                mat = factor
            else:
                mat = self.scipy_two_kfacs_to_mat(mat, factor)

        return mat

    def make_matrix_for_multiplication_with(self, kfac, cols=None):
        cols = cols if cols is not None else self.KFACMP_COLS
        assert bp_utils.is_matrix(kfac)
        _, rows = kfac.shape
        return torch.rand(rows, cols)

    def make_vector_for_multiplication_with(self, kfac):
        vec = self.make_matrix_for_multiplication_with(kfac, cols=1).squeeze(-1)
        assert bp_utils.is_vector(vec)
        return vec

    # TESTS
    ##########################################################################

    def test_two_kfacs_to_mat(self):
        """Check matrix from two Kronecker factors with `scipy`."""
        NUM_FACS = 2

        for _ in range(self.RUNS):
            A, B = self.make_random_kfacs(NUM_FACS)

            bp_result = bp_utils.two_kfacs_to_mat(A, B)
            sp_result = self.scipy_two_kfacs_to_mat(A, B)

            assert self.allclose(bp_result, sp_result)

    def test_kfacs_to_mat(self):
        """Check matrix from list of Kronecker factors with `scipy`."""
        for _ in range(self.RUNS):
            factors = self.make_random_kfacs()

            bp_result = bp_utils.kfacs_to_mat(factors)
            sp_result = self.scipy_kfacs_to_mat(factors)

            assert self.allclose(bp_result, sp_result)

    def test_apply_kfac_mat_prod(self):
        """Check matrix multiplication from Kronecker factors with matrix."""
        make_vec = self.make_vector_for_multiplication_with
        self.compare_kfac_tensor_prod(make_vec)

    def test_apply_kfac_vec_prod(self):
        """Check matrix multiplication from Kronecker factors with vector."""
        make_mat = self.make_matrix_for_multiplication_with
        self.compare_kfac_tensor_prod(make_mat)

    def compare_kfac_tensor_prod(self, make_tensor):
        def set_up():
            factors = self.make_random_kfacs()
            kfac = bp_utils.kfacs_to_mat(factors)
            tensor = make_tensor(kfac)
            return factors, kfac, tensor

        for _ in range(self.RUNS):
            factors, kfac, tensor = set_up()

            bp_result = bp_utils.apply_kfac_mat_prod(factors, tensor)
            torch_result = torch.matmul(kfac, tensor)

            assert self.allclose(bp_result, torch_result)

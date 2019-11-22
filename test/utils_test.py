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

    # HELPERS
    ##########################################################################
    def allclose(self, tensor1, tensor2):
        return torch.allclose(tensor1, tensor2, rtol=self.RTOL, atol=self.ATOL)

    def make_random_kfacs(self, num_facs=None):
        def random_kfac():
            def random_dim():
                return random.randint(self.MIN_DIM, self.MAX_DIM)

            shape = (random_dim(), random_dim())
            return torch.rand(shape)

        def random_num_facs():
            return random.randint(self.MIN_FACS, self.MAX_FACS)

        num_facs = num_facs if num_facs is not None else random_num_facs()
        return [random_kfac() for _ in range(num_facs)]

    # SCIPY implementations
    ##########################################################################

    def scipy_matrix_from_two_kron_facs(self, A, B):
        return torch.from_numpy(scipy.linalg.kron(A.numpy(), B.numpy()))

    def scipy_matrix_from_kron_facs(self, factors):
        mat = None
        for factor in factors:
            if mat is None:
                assert bp_utils.is_matrix(factor)
                mat = factor
            else:
                mat = self.scipy_matrix_from_two_kron_facs(mat, factor)

        return mat

    # TESTS
    ##########################################################################

    def test_matrix_from_two_kron_facs(self):
        """Check matrix from two Kronecker factors with `scipy`."""
        NUM_FACS = 2

        for _ in range(self.RUNS):
            A, B = self.make_random_kfacs(NUM_FACS)

            bp_result = bp_utils.matrix_from_two_kron_facs(A, B)
            sp_result = self.scipy_matrix_from_two_kron_facs(A, B)

            assert self.allclose(bp_result, sp_result)

    def test_matrix_from_kron_facs(self):
        """Check matrix from list of Kronecker factors with `scipy`."""
        for _ in range(self.RUNS):
            factors = self.make_random_kfacs()

            bp_result = bp_utils.matrix_from_kron_facs(factors)
            sp_result = self.scipy_matrix_from_kron_facs(factors)

            assert self.allclose(bp_result, sp_result)

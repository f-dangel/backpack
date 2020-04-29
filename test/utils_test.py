"""Test of Kronecker utilities."""

import random
import unittest

import scipy.linalg
import torch
from torch import einsum

from backpack.utils import kroneckers as bp_utils


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

    # Minimum eigenvalue of positive semi-definite
    PSD_KFAC_MIN_EIGVAL = 1

    # HELPERS
    ##########################################################################
    def allclose(self, tensor1, tensor2):
        return torch.allclose(tensor1, tensor2, rtol=self.RTOL, atol=self.ATOL)

    def list_allclose(self, tensor_list1, tensor_list2):
        assert len(tensor_list1) == len(tensor_list2)
        close = [self.allclose(t1, t2) for t1, t2 in zip(tensor_list1, tensor_list2)]
        print(close)
        for is_close, t1, t2 in zip(close, tensor_list1, tensor_list2):
            if not is_close:
                print(t1)
                print(t2)
        return all(close)

    def make_random_kfacs(self, num_facs=None):
        def random_kfac():
            def random_dim():
                return random.randint(self.MIN_DIM, self.MAX_DIM)

            shape = [random_dim(), random_dim()]
            return torch.rand(shape)

        def random_num_facs():
            return random.randint(self.MIN_FACS, self.MAX_FACS)

        num_facs = num_facs if num_facs is not None else random_num_facs()
        return [random_kfac() for _ in range(num_facs)]

    def make_random_psd_kfacs(self, num_facs=None):
        def make_quadratic_psd(mat):
            """Make matrix positive semi-definite: A -> AAᵀ."""
            mat_squared = einsum("ij,kj->ik", (mat, mat))
            shift = self.PSD_KFAC_MIN_EIGVAL * self.torch_eye_like(mat_squared)
            return mat_squared + shift

        kfacs = self.make_random_kfacs(num_facs=num_facs)
        return [make_quadratic_psd(fac) for fac in kfacs]

    # Torch helpers
    #########################################################################
    @staticmethod
    def torch_eye_like(tensor):
        return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

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

    def scipy_inv(self, mat, shift):
        mat_shifted = (shift * self.torch_eye_like(mat) + mat).numpy()
        inv = scipy.linalg.inv(mat_shifted)
        return torch.from_numpy(inv)

    def scipy_inv_kfacs(self, factors, shift_list):
        assert len(factors) == len(shift_list)
        return [self.scipy_inv(fac, shift) for fac, shift in zip(factors, shift_list)]

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

    def test_inv_kfacs(self):
        def get_shift():
            return random.random()

        for _ in range(self.RUNS):
            kfacs = self.make_random_psd_kfacs()
            num_kfacs = len(kfacs)

            # None vs 0.
            default_result = bp_utils.inv_kfacs(kfacs)
            no_shift_result = bp_utils.inv_kfacs(kfacs, shift=0.0)
            assert self.list_allclose(default_result, no_shift_result)

            # 0. vs tiny
            tiny = 1e-4
            tiny_shift_result = bp_utils.inv_kfacs(kfacs, shift=tiny)
            assert not self.list_allclose(no_shift_result, tiny_shift_result)

            # scalar vs. list of scalar: shift a should equal shift [a, a, ...]
            shift = get_shift()
            scalar_result = bp_utils.inv_kfacs(kfacs, shift=shift)
            list_result = bp_utils.inv_kfacs(kfacs, shift=num_kfacs * [shift])
            assert self.list_allclose(scalar_result, list_result)

            # scipy vs. torch
            shift_list = [get_shift() for _ in range(num_kfacs)]
            bp_result = bp_utils.inv_kfacs(kfacs, shift=shift_list)
            sp_result = self.scipy_inv_kfacs(kfacs, shift_list)
            assert self.list_allclose(bp_result, sp_result)

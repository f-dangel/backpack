"""Test of Kronecker utilities."""

import random

import torch

import scipy.linalg
from backpack.extensions.secondorder import utils as bp_utils


# HELPERS
##############################################################################
def scipy_matrix_from_two_kron_facs(A, B):
    return torch.from_numpy(scipy.linalg.kron(A.numpy(), B.numpy()))


def scipy_matrix_from_kron_facs(factors):
    mat = None
    for factor in factors:
        if mat is None:
            assert bp_utils.is_matrix(factor)
            mat = factor
        else:
            mat = scipy_matrix_from_two_kron_facs(mat, factor)

    return mat


def random_matrix(min_dim=1, max_dim=10):
    def random_int():
        return random.randint(min_dim, max_dim)

    shape = (random_int(), random_int())
    return torch.rand(shape)


##############################################################################
ATOL = 1e-6
RTOL = 1e-5


def test_matrix_from_two_kron_facs(random_runs=100, min_dim=1, max_dim=10):
    """Check matrix from two Kronecker factors with `scipy`."""

    def make_factors():
        A = random_matrix(min_dim=min_dim, max_dim=max_dim)
        B = random_matrix(min_dim=min_dim, max_dim=max_dim)
        return A, B

    for _ in range(random_runs):
        A, B = make_factors()

        bp_result = bp_utils.matrix_from_two_kron_facs(A, B)
        sp_result = scipy_matrix_from_two_kron_facs(A, B)

        assert torch.allclose(bp_result, sp_result, atol=ATOL, rtol=RTOL)


def test_matrix_from_kron_facs(
    runs=100, min_num=1, max_num=3, min_dim=1, max_dim=10,
):
    """Check matrix from list of Kronecker factors with `scipy`."""

    def make_factors():
        factors = []
        num_factors = random.randint(min_num, max_num)

        def random_dim():
            return random.randint(min_dim, max_dim)

        for _ in range(num_factors):
            shape = (random_dim(), random_dim())
            factors.append(torch.rand(shape))

        return factors

    for _ in range(runs):
        factors = make_factors()

        bp_result = bp_utils.matrix_from_kron_facs(factors)
        sp_result = scipy_matrix_from_kron_facs(factors)

        assert torch.allclose(bp_result, sp_result, atol=ATOL, rtol=RTOL)

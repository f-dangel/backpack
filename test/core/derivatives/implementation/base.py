"""Contains DerivativesImplementation, the base class for autograd and backpack."""
from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class DerivativesImplementation(ABC):
    """Base class for autograd and BackPACK implementations."""

    def __init__(self, problem):
        """Initialization.

        Args:
            problem: test problem
        """
        self.problem = problem

    @abstractmethod
    def jac_mat_prod(self, mat: Tensor) -> Tensor:
        """Vectorized product of input-output-Jacobian and a matrix.

        Args:
            mat: matrix: the vectors along its leading dimension will be multiplied.

        Returns:
            Tensor representing the result of Jacobian-vector product.
                product[v] = J @ mat[v]
        """
        raise NotImplementedError

    @abstractmethod
    def jac_t_mat_prod(self, mat: Tensor, subsampling: List[int] = None) -> Tensor:
        """Vectorized product of transposed jacobian and matrix.

        Args:
            mat: matrix: the vectors along its leading dimension will be multiplied.
            subsampling: Active samples in the output. Default: ``None`` (all).

        Returns:
            Tensor representing the result of Jacobian-vector product.
                product[v] = mat[v] @ J
        """
        raise NotImplementedError

    @abstractmethod
    def param_mjp(
        self,
        param_str: str,
        mat: Tensor,
        sum_batch: bool,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Matrix-Jacobian products w.r.t. the parameter.

        Args:
            param_str: parameter name
            mat: matrix
            sum_batch: whether to sum along batch axis
            subsampling: Active samples in the output. Default: ``None`` (all).

        Returns:
            product
        """
        raise NotImplementedError

    @abstractmethod
    def weight_jac_mat_prod(self, mat: Tensor) -> Tensor:
        """Product of jacobian and matrix.

        Args:
            mat: matrix

        Returns:
            product
        """
        raise NotImplementedError

    @abstractmethod
    def bias_jac_mat_prod(self, mat: Tensor) -> Tensor:
        """Product of jacobian and matrix.

        Args:
            mat: matrix

        Returns:
            product
        """
        raise NotImplementedError

    @abstractmethod
    def ea_jac_t_mat_jac_prod(self, mat: Tensor) -> Tensor:
        """Product of ea jacobian with matrix.

        Args:
            mat: matrix

        Returns:
            product
        """
        raise NotImplementedError

    @abstractmethod
    def sum_hessian(self) -> Tensor:
        """Sum of hessians.

        Returns:
            the sum of hessians
        """
        raise NotImplementedError

    @abstractmethod
    def hessian_is_zero(self) -> bool:
        """Return whether the input-output Hessian is zero.

        Returns:
            `True`, if Hessian is zero, else `False`.
        """
        raise NotImplementedError

    @abstractmethod
    def hessian_mat_prod(self, mat: Tensor) -> Tensor:
        """Product of hessian with matrix mat.

        Args:
            mat: matrix to multiply

        Returns:
            product
        """
        raise NotImplementedError

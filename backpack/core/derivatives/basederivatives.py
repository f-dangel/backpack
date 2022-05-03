"""Base classes for more flexible Jacobians and second-order information."""
import warnings
from abc import ABC
from typing import Callable, List, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives import shape_check


class BaseDerivatives(ABC):
    """First- and second-order partial derivatives of unparameterized module.

    Note:
        Throughout the code, use these conventions if possible:

        - `N`: batch size
        - Vectors
          - Layer input shape `[N, D_in]`
          - Layer output shape `[N, D_out]`
        - Images
          - Layer input shape `[N, C_in, H_in, W_in]`
          - Layer output shape `[N, C_out, H_out, W_out]`
        - `V`: vectorization axis

    Definition:
        For simplicity, consider the vector case, i.e. a function which maps an
        `[N, D_in]` `input` into an `[N, D_out]` `output`.

        The input-output Jacobian `J` of  is tensor of shape `[N, D_out, N_in, D_in]`.
        Partial derivatives are ordered as

            `J[i, j, k, l] = 𝜕output[i, j] / 𝜕input[k, l].

        The transposed input-output Jacobian `Jᵀ` has shape `[N, D_in, N, D_out]`.
        Partial derivatives are ordered as

            `Jᵀ[i, j, k, l] = 𝜕output[k, l] / 𝜕input[i, j]`.

        In general, feature dimension indices `j, l` are product indices.
    """

    @shape_check.jac_mat_prod_accept_vectors
    @shape_check.jac_mat_prod_check_shapes
    def jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Apply Jacobian of the output w.r.t. input to a matrix.

        It is assumed that the module input has shape `[N, *]`, while the output is
        of shape `[N, •]`. Both `*`, `•` denote arbitrary shapes.

        Apply Jacobian to all slices among the vectorization axis.

            `result[v, n, •] =  ∑ₖ ∑_* J[n, •, k, *] mat[v, n, *]`.

        Args:
            module: Extended module.
            g_inp: Gradients of the module w.r.t. its inputs.
            g_out: Gradients of the module w.r.t. its outputs.
            mat: Matrix the Jacobian will be applied to. Must have
                shape `[V, N, *]`.

        Returns:
            Jacobian-matrix product. Has shape [V, N, *].

        Note:
            - The Jacobian can be applied without knowledge about backpropagated
              derivatives. Both `g_inp` and `g_out` are usually not required and
              can be set to `None`.
        """
        return self._jac_mat_prod(module, g_inp, g_out, mat)

    def _jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @shape_check.jac_t_mat_prod_accept_vectors
    @shape_check.jac_t_mat_prod_check_shapes
    def jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Apply transposed input-ouput Jacobian of module output to a matrix.

        Implicit application of Jᵀ:
            result[v, ̃n, ̃c, ̃w, ...]
            = ∑_{n, c, w} Jᵀ[̃n, ̃c, ̃w, ..., n, c, w, ...] mat[v, n, c, w, ...].

        Args:
            module: module which derivative is calculated
            g_inp: input gradients
            g_out: output gradients
            mat: Matrix the transposed Jacobian will be applied to.
                Must have shape ``[V, *module.output.shape]``; but if used with
                sub-sampling, the batch dimension is replaced by ``len(subsampling)``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Transposed Jacobian-matrix product.
            Has shape ``[V, *module.input0.shape]``; but if used with sub-sampling,
            the batch dimension is replaced by ``len(subsampling)``.
        """
        return self._jac_t_mat_prod(module, g_inp, g_out, mat, subsampling=subsampling)

    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        raise NotImplementedError

    # TODO Add shape check
    # TODO Use new convention
    def ea_jac_t_mat_jac_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Expectation approximation of outer product with input-output Jacobian.

        Used for backpropagation in KFRA.

        For `yₙ = f(xₙ) n=1,...,n`, compute `E(Jₙᵀ mat Jₙ) = 1/n ∑ₙ Jₙᵀ mat Jₙ`.
        In index notation, let `output[n]=f(input[n]) n = 1,...,n`. Then,
        `result[i,j]
        = 1/n ∑ₙₖₗ (𝜕output[n,k] / 𝜕input[n,i]) mat[k,l] (𝜕output[n,j] / 𝜕input[n,l])

        Args:
            module: Extended module.
            g_inp: Gradients of the module w.r.t. its inputs.
            g_out: Gradients of the module w.r.t. its outputs.
            mat: Matrix of shape `[D_out, D_out]`.

        # noqa: DAR202
        Returns:
            Matrix of shape `[D_in, D_in]`.

        Note:
            - This operation can be applied without knowledge about backpropagated
              derivatives. Both `g_inp` and `g_out` are usually not required and
              can be set to `None`.

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError

    def hessian_is_zero(self, module: Module) -> bool:
        """Returns whether Hessian is zero.

        I.e. whether ``∂²output[i] / ∂input[j] ∂input[k] = 0  ∀ i,j,k``.

        Args:
            module: current module to evaluate

        # noqa: DAR202
        Returns:
            whether Hessian is zero

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError

    def hessian_is_diagonal(self, module: Module) -> bool:
        """Is `∂²output[i] / ∂input[j] ∂input[k]` nonzero only if `i = j = k`.

        The Hessian diagonal is only defined for layers that preserve the size
        of their input.

        Must be implemented by descendants that don't implement ``hessian_is_zero``.

        Args:
            module: current module to evaluate

        # noqa: DAR202
        Returns:
            whether Hessian is diagonal

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError

    # FIXME Currently returns `∂²output[i] / ∂input[i]² * g_out[0][i]`,
    # which s the residual matrix diagonal, rather than the Hessian diagonal
    def hessian_diagonal(
        self, module: Module, g_in: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Tensor:
        """Return the Hessian diagonal `∂²output[i] / ∂input[i]²`.

        Only required if `hessian_is_diagonal` returns `True`.
        The Hessian diagonal is only defined for layers that preserve the size
        of their input.

        Args:
            module: Module whose output-input Hessian diagonal is computed.
            g_in: Gradients w.r.t. the module input.
            g_out: Gradients w.r.t. the module output.

        # noqa: DAR202
        Returns:
            Hessian diagonal. Has same shape as module input.

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError

    def hessian_is_psd(self) -> bool:
        """Is `∂²output[i] / ∂input[j] ∂input[k]` positive semidefinite (PSD).

        # noqa: DAR202
        Returns:
            whether hessian is positive semi definite

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError

    @shape_check.residual_mat_prod_accept_vectors
    @shape_check.residual_mat_prod_check_shapes
    def residual_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Multiply with the residual term.

        Performs mat → [∑_{k} Hz_k(x) 𝛿z_k] mat.

        Args:
            module: module
            g_inp: input gradients
            g_out: output gradients
            mat: matrix to multiply

        Returns:
            product

        Note:
            This function only has to be implemented if the residual is not
            zero and not diagonal (for instance, `BatchNorm`).
        """
        return self._residual_mat_prod(module, g_inp, g_out, mat)

    def _residual_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _reshape_like(mat: Tensor, shape: Tuple[int]) -> Tensor:
        """Reshape as like with trailing and additional 0th dimension.

        If like is [N, C, H, ...], returns shape [-1, N, C, H, ...]

        Args:
            mat: Matrix to reshape.
            shape: Trailing target shape.

        Returns:
            reshaped matrix
        """
        return mat.reshape(-1, *shape)

    @classmethod
    def reshape_like_input(
        cls, mat: Tensor, module: Module, subsampling: List[int] = None
    ) -> Tensor:
        """Reshapes matrix according to input.

        Args:
            mat: matrix to reshape
            module: module which input shape is used
            subsampling: Indices of active samples. ``None`` means use all samples.

        Returns:
            reshaped matrix
        """
        shape = list(module.input0.shape)
        if subsampling is not None:
            shape[0] = len(subsampling)

        return cls._reshape_like(mat, shape)

    @classmethod
    def reshape_like_output(cls, mat: Tensor, module: Module) -> Tensor:
        """Reshapes matrix like output.

        Args:
            mat: matrix to reshape
            module: module which output is used

        Returns:
            reshaped matrix
        """
        return cls._reshape_like(mat, module.output.shape)


class BaseParameterDerivatives(BaseDerivatives, ABC):
    """First- and second order partial derivatives of a module with parameters.

    Assumptions (true for `nn.Linear`, `nn.Conv(Transpose)Nd`, `nn.BatchNormNd`):
    - Parameters are saved as `.weight` and `.bias` fields in a module
    - The output is linear in the model parameters

    Shape conventions:
    ------------------
    Weight [C_w, H_w, W_w, ...] (usually 1d, 2d, 4d)
    Bias [C_b, ...] (usually 1d)

    For most layers, these shapes correspond to shapes of the module input or output.
    """

    @shape_check.param_mjp_accept_vectors
    def param_mjp(
        self,
        param_str: str,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Compute matrix-Jacobian products (MJPs) of the module w.r.t. a parameter.

        Handles both vector and matrix inputs. Preserves input format in output.

        Internally calls out to ``_{param_str}_jac_t_mat_prod`` function that must be
        implemented by descendants. It follows the same signature, but does not have
        the ``param_str`` argument.

        Args:
            param_str: Attribute name under which the parameter is stored in the module.
            module: Module whose Jacobian will be applied. Must provide access to IO.
            g_inp: Gradients w.r.t. module input.
            g_out: Gradients w.r.t. module output.
            mat: Matrix the Jacobian will be applied to. Has shape
                ``[V, *module.output.shape]`` (matrix case) or same shape as
                ``module.output`` (vector case). If used with subsampling, has dimension
                len(subsampling) instead of batch size along the batch axis.
            sum_batch: Sum out the MJP's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Matrix-Jacobian products. Has shape ``[V, *param_shape]`` when batch
            summation is enabled (same shape as parameter in the vector case). Without
            batch summation, the result has shape ``[V, N, *param_shape]`` (vector case
            has shape ``[N, *param_shape]``). If used with subsampling, the batch size N
            is replaced by len(subsampling).

        Raises:
            NotImplementedError: if required method is not implemented by derivatives class
        """
        # input check
        shape_check.shape_like_output(mat, module, subsampling=subsampling)

        method_name = f"_{param_str}_jac_t_mat_prod"
        mjp = getattr(self, method_name, None)
        if mjp is None:
            raise NotImplementedError(
                f"Computation requires implementation of {method_name}, but {self} "
                f"(defining derivatives of {module}) does not implement it."
            )
        mjp_out = mjp(
            module, g_inp, g_out, mat, sum_batch=sum_batch, subsampling=subsampling
        )

        # output check
        shape_check.check_like_with_sum_batch(
            mjp_out, module, param_str, sum_batch=sum_batch
        )
        shape_check.check_same_V_dim(mjp_out, mat)

        return mjp_out

    @shape_check.bias_jac_mat_prod_accept_vectors
    @shape_check.bias_jac_mat_prod_check_shapes
    def bias_jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Apply Jacobian of the output w.r.t. bias to a matrix.

        Args:
            module: module to perform derivatives on
            g_inp: input gradients
            g_out: output gradients
            mat: Matrix the Jacobian will be applied to.
                Must have shape [V, C_b, ...].

        Returns:
            Jacobian-matrix product. Has shape [V, N, C_out, H_out, ...].
        """
        return self._bias_jac_mat_prod(module, g_inp, g_out, mat)

    def _bias_jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @shape_check.weight_jac_mat_prod_accept_vectors
    @shape_check.weight_jac_mat_prod_check_shapes
    def weight_jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Apply Jacobian of the output w.r.t. weight to a matrix.

        Args:
            module: module to perform derivatives on
            g_inp: input gradients
            g_out: output gradients
            mat: Matrix the Jacobian will be applied to.
                Must have shape [V, C_w, H_w, ...].

        Returns:
            Jacobian-matrix product.
            Has shape [V, N, C_out, H_out, ...].
        """
        return self._weight_jac_mat_prod(module, g_inp, g_out, mat)

    def _weight_jac_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        raise NotImplementedError


class BaseLossDerivatives(BaseDerivatives, ABC):
    """Second- order partial derivatives of loss functions."""

    # TODO Add shape check
    def sqrt_hessian(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        """Symmetric factorization ('sqrt') of the loss Hessian.

        The Hessian factorization is returned in format ``Hs = [D, N, D]``, where
        ``Hs[:, n, :]`` is the Hessian factorization for the ``n``th sample, i.e.
        ``Hs[:, n, :]ᵀ Hs[:, n, :]`` is the Hessian w.r.t. to the ``n``th sample.

        Args:
            module: Loss layer whose factorized Hessian will be computed.
            g_inp: Gradients w.r.t. module input.
            g_out: Gradients w.r.t. module output.
            subsampling: Indices of data samples to be considered. Default of ``None``
                uses all data in the mini-batch.

        Returns:
            Symmetric factorization of the loss Hessian for each sample. If the input
            to the loss has shape ``[N, D]``, this is a tensor of shape ``[D, N, D]``;
            if used with sub-sampling, ``N`` is replaced by ``len(subsampling)``.
            For fixed ``n``, squaring the matrix implied by the slice ``[:, n, :]``
            results in the loss Hessian w.r.t. to sample ``n``.
        """
        self._check_2nd_order_make_sense(module, g_out)
        return self._sqrt_hessian(module, g_inp, g_out, subsampling=subsampling)

    def _sqrt_hessian(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        raise NotImplementedError

    # TODO Add shape check
    def sqrt_hessian_sampled(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:
        """A Monte-Carlo sampled symmetric factorization of the loss Hessian.

        The Hessian factorization is returned in format ``Hs = [M, N, D]``, where
        ``Hs[:, n, :]`` approximates the Hessian factorization for the ``n``th sample,
        i.e. ``Hs[:, n, :]ᵀ Hs[:, n, :]ᵀ`` approximates the Hessian w.r.t. to sample
        ``n``.

        Args:
            module: Loss layer whose factorized Hessian will be computed.
            g_inp: Gradients w.r.t. module input.
            g_out: Gradients w.r.t. module output.
            mc_samples: Number of samples used for MC approximation.
            subsampling: Indices of data samples to be considered. Default of ``None``
                uses all data in the mini-batch.

        Returns:
            Symmetric factorization of the loss Hessian for each sample. If the input
            to the loss has shape ``[N, D]``, this is a tensor of shape ``[M, N, D]``
            when using ``M`` MC samples; if used with sub-sampling, ``N`` is replaced
            by ``len(subsampling)``. For fixed ``n``, squaring the matrix implied by the
            slice ``[:, n, :]`` approximates the loss Hessian w.r.t. to sample ``n``.
        """
        self._check_2nd_order_make_sense(module, g_out)
        return self._sqrt_hessian_sampled(
            module, g_inp, g_out, mc_samples=mc_samples, subsampling=subsampling
        )

    def _sqrt_hessian_sampled(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling=None,
    ) -> Tensor:
        raise NotImplementedError

    @shape_check.make_hessian_mat_prod_accept_vectors
    @shape_check.make_hessian_mat_prod_check_shapes
    def make_hessian_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Callable[[Tensor], Tensor]:
        """Multiplication of the input Hessian with a matrix.

        Return a function that maps mat to H * mat.

        Args:
            module: module to perform derivatives on
            g_inp: input gradients
            g_out: output gradients

        Returns:
            function that maps mat to H * mat
        """
        self._check_2nd_order_make_sense(module, g_out)
        return self._make_hessian_mat_prod(module, g_inp, g_out)

    def _make_hessian_mat_prod(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Callable[[Tensor], Tensor]:
        raise NotImplementedError

    # TODO Add shape check
    def sum_hessian(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Tensor:
        """Loss Hessians, summed over the batch dimension.

        Args:
            module: module to perform derivatives on
            g_inp: input gradients
            g_out: output gradients

        Returns:
            sum of hessians
        """
        self._check_2nd_order_make_sense(module, g_out)
        return self._sum_hessian(module, g_inp, g_out)

    def _sum_hessian(
        self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Tensor:
        raise NotImplementedError

    def _check_2nd_order_make_sense(self, module: Module, g_out: Tuple[Tensor]) -> None:
        """Verify conditions for 2nd-order extensions to be working.

        2nd-order extensions are only guaranteed to work if the `loss`,
        on which `backward()` is called, is a scalar that has not been
        modified further after passing through the loss function module.

        Args:
            module: module to perform derivatives on
            g_out: output gradients
        """
        self._check_output_is_scalar(module)
        self._check_loss_has_not_been_modified(module, g_out)

    @classmethod
    def _check_output_is_scalar(cls, module: Module) -> None:
        """Raise an exception is the module output is not a scalar.

        Args:
            module: module to perform derivatives on

        Raises:
            ValueError: if output is not scalar
        """
        if module.output.numel() != 1:
            raise ValueError(
                "Output must be scalar. Got {}".format(module.output.shape)
            )

    @classmethod
    def _check_loss_has_not_been_modified(
        cls, module: Module, g_out: Tuple[Tensor]
    ) -> None:
        """Raise a warning if the module output seems to have been changed.

        Args:
            module: module to perform derivatives on
            g_out: output gradients
        """
        grad_out_is_identity = g_out is None or (g_out[0] == 1.0).all().item()
        if not grad_out_is_identity:
            warnings.warn(
                "The output of {} seems to have been modified.".format(module)
                + " Backpack might give wrong second-order information."
                + " Make sure you call backward() on the output of a loss"
                + " function module from torch.nn",
                UserWarning,
            )

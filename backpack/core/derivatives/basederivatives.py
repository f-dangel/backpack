"""Base classes for more flexible Jacobians and second-order information."""
import warnings

from backpack.core.derivatives import shape_check


class BaseDerivatives:
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
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. input to a matrix.

        It is assumed that the module input has shape `[N, *]`, while the output is
        of shape `[N, •]`. Both `*`, `•` denote arbitrary shapes.

        Apply Jacobian to all slices among the vectorization axis.

            `result[v, n, •] =  ∑ₖ ∑_* J[n, •, k, *] mat[v, n, *]`.

        Args:
            module (torch.nn.Module): Extended module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.
            mat (torch.Tensor): Matrix the Jacobian will be applied to. Must have
                shape `[V, N, *]`.

        Returns:
            torch.Tensor: Jacobian-matrix product. Has shape [V, N, *].

        Note:
            - The Jacobian can be applied without knowledge about backpropagated
              derivatives. Both `g_inp` and `g_out` are usually not required and
              can be set to `None`.
        """
        return self._jac_mat_prod(module, g_inp, g_out, mat)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        """Internal implementation of the input-output Jacobian."""
        raise NotImplementedError

    @shape_check.jac_t_mat_prod_accept_vectors
    @shape_check.jac_t_mat_prod_check_shapes
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """Apply transposed input-ouput Jacobian of module output to a matrix.

        Implicit application of Jᵀ:
            result[v, ̃n, ̃c, ̃w, ...]
            = ∑_{n, c, w} Jᵀ[̃n, ̃c, ̃w, ..., n, c, w, ...] mat[v, n, c, w, ...].

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the transposed Jacobian will be applied to.
            Must have shape [V, N, C_out, H_out, ...].

        Returns:
        --------
        result: torch.Tensor
            Transposed Jacobian-matrix product.
            Has shape [V, N, C_in, H_in, ...].
        """
        return self._jac_t_mat_prod(module, g_inp, g_out, mat)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """Internal implementation of transposed Jacobian."""
        raise NotImplementedError

    # TODO Add shape check
    # TODO Use new convention
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """Expectation approximation of outer product with input-output Jacobian.

        Used for backpropagation in KFRA.

        For `yₙ = f(xₙ) n=1,...,n`, compute `E(Jₙᵀ mat Jₙ) = 1/n ∑ₙ Jₙᵀ mat Jₙ`.
        In index notation, let `output[n]=f(input[n]) n = 1,...,n`. Then,
        `result[i,j]
        = 1/n ∑ₙₖₗ (𝜕output[n,k] / 𝜕input[n,i]) mat[k,l] (𝜕output[n,j] / 𝜕input[n,l])

        Args:
            module (torch.nn.Module): Extended module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.
            mat (torch.Tensor): Matrix of shape `[D_out, D_out]`.

        Returns:
            torch.Tensor: Matrix of shape `[D_in, D_in]`.

        Note:
            - This operation can be applied without knowledge about backpropagated
              derivatives. Both `g_inp` and `g_out` are usually not required and
              can be set to `None`.
        """
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        """Is `∂²output[i] / ∂input[j] ∂input[k]` nonzero only if `i = j = k`."""
        raise NotImplementedError

    def hessian_diagonal(self):
        """Return `∂²output[i] / ∂input[i]²`.

        Only required if `hessian_is_diagonal` returns `True`.
        """
        raise NotImplementedError

    def hessian_is_psd(self):
        """Is `∂²output[i] / ∂input[j] ∂input[k]` positive semidefinite (PSD)."""
        raise NotImplementedError

    @shape_check.residual_mat_prod_accept_vectors
    @shape_check.residual_mat_prod_check_shapes
    def residual_mat_prod(self, module, g_inp, g_out, mat):
        """Multiply with the residual term.

        Performs mat → [∑_{k} Hz_k(x) 𝛿z_k] mat.

        Note:
        -----
            This function only has to be implemented if the residual is not
            zero and not diagonal (for instance, `BatchNorm`).
        """
        return self._residual_mat_prod(module, g_inp, g_out, mat)

    def _residual_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    @staticmethod
    def _reshape_like(mat, like):
        """Reshape as like with trailing and additional 0th dimension.

        If like is [N, C, H, ...], returns shape [-1, N, C, H, ...]
        """
        V = -1
        shape = (V, *like.shape)
        return mat.reshape(shape)

    @classmethod
    def reshape_like_input(cls, mat, module):
        return cls._reshape_like(mat, module.input0)

    @classmethod
    def reshape_like_output(cls, mat, module):
        return cls._reshape_like(mat, module.output)


class BaseParameterDerivatives(BaseDerivatives):
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

    @shape_check.bias_jac_mat_prod_accept_vectors
    @shape_check.bias_jac_mat_prod_check_shapes
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. bias to a matrix.

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the Jacobian will be applied to.
            Must have shape [V, C_b, ...].

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_out, H_out, ...].
        """
        return self._bias_jac_mat_prod(module, g_inp, g_out, mat)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Internal implementation of the bias Jacobian."""
        raise NotImplementedError

    @shape_check.bias_jac_t_mat_prod_accept_vectors
    @shape_check.bias_jac_t_mat_prod_check_shapes
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. bias to a matrix.

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the transposed Jacobian will be applied to.
            Must have shape [V, N, C_out, H_out, ...].
        sum_batch: bool
            Whether to sum over the batch dimension on the fly.

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_b, ...] if `sum_batch == False`.
            Has shape [V, C_b, ...] if `sum_batch == True`.
        """
        return self._bias_jac_t_mat_prod(module, g_inp, g_out, mat, sum_batch=sum_batch)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Internal implementation of the transposed bias Jacobian."""
        raise NotImplementedError

    @shape_check.weight_jac_mat_prod_accept_vectors
    @shape_check.weight_jac_mat_prod_check_shapes
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. weight to a matrix.

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the Jacobian will be applied to.
            Must have shape [V, C_w, H_w, ...].

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_out, H_out, ...].
        """
        return self._weight_jac_mat_prod(module, g_inp, g_out, mat)

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Internal implementation of weight Jacobian."""
        raise NotImplementedError

    @shape_check.weight_jac_t_mat_prod_accept_vectors
    @shape_check.weight_jac_t_mat_prod_check_shapes
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. weight to a matrix.

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the transposed Jacobian will be applied to.
            Must have shape [V, N, C_out, H_out, ...].
        sum_batch: bool
            Whether to sum over the batch dimension on the fly.

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_w, H_w, ...] if `sum_batch == False`.
            Has shape [V, C_w, H_w, ...] if `sum_batch == True`.
        """
        return self._weight_jac_t_mat_prod(
            module, g_inp, g_out, mat, sum_batch=sum_batch
        )

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Internal implementation of transposed weight Jacobian."""
        raise NotImplementedError


class BaseLossDerivatives(BaseDerivatives):
    """Second- order partial derivatives of loss functions."""

    # TODO Add shape check
    def sqrt_hessian(self, module, g_inp, g_out):
        """Symmetric factorization ('sqrt') of the loss Hessian."""
        self.check_2nd_order_make_sense(module, g_inp, g_out)
        return self._sqrt_hessian(module, g_inp, g_out)

    def _sqrt_hessian(self, module, g_inp, g_out):
        raise NotImplementedError

    # TODO Add shape check
    def sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        """Monte-Carlo sampled symmetric factorization of the loss Hessian."""
        self.check_2nd_order_make_sense(module, g_inp, g_out)
        return self._sqrt_hessian_sampled(module, g_inp, g_out, mc_samples=mc_samples)

    def _sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        raise NotImplementedError

    @shape_check.make_hessian_mat_prod_accept_vectors
    @shape_check.make_hessian_mat_prod_check_shapes
    def make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix.

        Return a function that maps mat to H * mat.
        """
        self.check_2nd_order_make_sense(module, g_inp, g_out)
        return self._make_hessian_mat_prod(module, g_inp, g_out)

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        raise NotImplementedError

    # TODO Add shape check
    def sum_hessian(self, module, g_inp, g_out):
        """Loss Hessians, summed over the batch dimension."""
        self.check_2nd_order_make_sense(module, g_inp, g_out)
        return self._sum_hessian(module, g_inp, g_out)

    def _sum_hessian(self, module, g_inp, g_out):
        raise NotImplementedError

    def check_2nd_order_make_sense(self, module, g_inp, g_out):
        """Verify conditions for 2nd-order extensions to be working.

        2nd-order extensions are only guaranteed to work if the `loss`,
        on which `backward()` is called, is a scalar that has not been
        modified further after passing through the loss function module.
        """
        self._check_output_is_scalar(module)
        self._check_loss_has_not_been_modified(module, g_out)

    def _check_output_is_scalar(self, module):
        """Raise an exception is the module output is not a scalar."""
        if module.output.numel() != 1:
            raise ValueError(
                "Output must be scalar. Got {}".format(module.output.shape)
            )

    def _check_loss_has_not_been_modified(self, module, g_out):
        """Raise a warning if the module output seems to have been changed."""
        grad_out_is_identity = g_out is None or (g_out[0] == 1.0).all().item()
        if not grad_out_is_identity:
            warnings.warn(
                "The output of {} seems to have been modified.".format(module)
                + " Backpack might give wrong second-order information."
                + " Make sure you call backward() on the output of a loss"
                + " function module from torch.nn",
                UserWarning,
            )

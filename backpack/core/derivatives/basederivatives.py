from backpack.core.derivatives import shape_check


class BaseDerivatives:
    """First- and second-order partial derivatives of a module.

    Shape conventions:
    ------------------
    * Batch size: N
    * Free dimension for vectorization: V

    For vector-processing layers (2d input):
    * input [N, C_in],  output [N, C_out]

    For image-processing layers (4d input)
    * Input/output channels: C_in/C_out
    * Input/output height: H_in/H_out
    * Input/output width: W_in/W_out
    * input [N, C_in, H_in, W_in],  output [N, C_out, H_in, W_in]


    Definitions:
    ------------
    * The Jacobian J is defined as
        J[n, c, w, ..., ̃n, ̃c, ̃w, ...]
        = 𝜕output[n, c, w, ...] / 𝜕input[̃n, ̃c, ̃w, ...]
    * The transposed Jacobian Jᵀ is defined as
        Jᵀ[̃n, ̃c, ̃w, ..., n, c, w, ...]
        = 𝜕output[n, c, w, ...] / 𝜕input[̃n, ̃c, ̃w, ...]
    """

    @shape_check.jac_mat_prod_accept_vectors
    @shape_check.jac_mat_prod_check_shapes
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. input to a matrix.

        Implicit application of J:
            result[v, n, c, w, ...]
            =  ∑_{̃n, ̃c, ̃w} J[n, c, w, ..., ̃n, ̃c, ̃w, ...] mat[̃n, ̃c, ̃w, ...].

        Arguments:
            module (torch.nn.Module): Extended module that has been used in a
                `forward` pass.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.
            mat (torch.Tensor): Matrix the Jacobian will be applied to. Must have
                shape [V, N, C_in, H_in, ...].

        Notes:
            - The Jacobian can be applied without knowledge about backpropagated
              derivatives. Both `g_inp` and `g_out` are usually not required and
              can be set to `None`.
            - It is currently unclear whether the additional information about the
              computational graph, encoded in `g_inp`, `g_out` might become relevant.

        Returns:
            torch.Tensor: Jacobian-matrix product.
                Has shape [V, N, C_out, H_out, ...].
        """
        return self._jac_mat_prod(module, g_inp, g_out, mat)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        """Internal implementation of the Jacobian."""
        raise NotImplementedError

    @shape_check.jac_t_mat_prod_accept_vectors
    @shape_check.jac_t_mat_prod_check_shapes
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """Apply transposed Jacobian of module output w.r.t. input to a matrix.

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
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        raise NotImplementedError

    def hessian_diagonal(self):
        raise NotImplementedError

    def hessian_is_psd(self):
        raise NotImplementedError

    # TODO make accept vectors
    # TODO add shape check
    def make_residual_mat_prod(self, module, g_inp, g_out):
        """Return multiplication routine with the residual term.

        The function performs the mapping: mat → [∑_{k} Hz_k(x) 𝛿z_k] mat.
        (required for extension `curvmatprod`)

        Note:
        -----
            This function only has to be implemented if the residual is not
            zero and not diagonal (for instance, `BatchNorm`).
        """
        raise NotImplementedError

    # TODO Refactor and remove
    def batch_flat(self, tensor):
        batch = tensor.size(0)
        # TODO Removing the clone().detach() will destroy the computation graph
        # Tests will fail
        return batch, tensor.clone().detach().view(batch, -1)

    # TODO Refactor and remove
    def get_batch(self, module):
        return module.input0.size(0)

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
    """Second- order partial derivatives of loss functions.

    """

    # TODO Add shape check
    def sqrt_hessian(self, module, g_inp, g_out):
        """Symmetric factorization ('sqrt') of the loss Hessian."""
        return self._sqrt_hessian(module, g_inp, g_out)

    def _sqrt_hessian(self, module, g_inp, g_out):
        raise NotImplementedError

    # TODO Add shape check
    def sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        """Monte-Carlo sampled symmetric factorization of the loss Hessian."""
        return self._sqrt_hessian_sampled(module, g_inp, g_out, mc_samples=mc_samples)

    def _sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        raise NotImplementedError

    @shape_check.make_hessian_mat_prod_accept_vectors
    @shape_check.make_hessian_mat_prod_check_shapes
    def make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix.

        Return a function that maps mat to H * mat.
        """
        return self._make_hessian_mat_prod(module, g_inp, g_out)

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        raise NotImplementedError

    # TODO Add shape check
    def sum_hessian(self, module, g_inp, g_out):
        """Loss Hessians, summed over the batch dimension."""
        return self._sum_hessian(module, g_inp, g_out)

    def _sum_hessian(self, module, g_inp, g_out):
        raise NotImplementedError

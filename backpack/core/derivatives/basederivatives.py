from backpack.utils.einsum import try_view


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

    MC_SAMPLES = 1

    def jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. input to a matrix.

        Implicit application of J:
            result[v, n, c, w, ...]
            =  ∑_{̃n, ̃c, ̃w} J[n, c, w, ..., ̃n, ̃c, ̃w, ...] mat[̃n, ̃c, ̃w, ...].
        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the transposed Jacobian will be applied to.
            Must have shape [V, N, C_in, H_in, ...].

        Returns:
        --------
        result: torch.Tensor
            Transposed-Jacobian-matrix product.
            Has shape [V, N, C_out, H_out, ...].
        """
        raise NotImplementedError

    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """Apply transposed Jacobian of module output w.r.t. input to a matrix.

        Implicit application of Jᵀ:
            result[v, ̃n, ̃c, ̃w, ...]
            = ∑_{n, c, w} Jᵀ[̃n, ̃c, ̃w, ..., n, c, w, ...] mat[v, n, c, w, ...].

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the Jacobian will be applied to.
            Must have shape [V, N, C_out, H_out, ...].

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_in, H_in, ...].
        """

        raise NotImplementedError

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        # TODO: Use new convention
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        raise NotImplementedError

    def hessian_diagonal(self):
        raise NotImplementedError

    def hessian_is_psd(self):
        raise NotImplementedError

    def batch_flat(self, tensor):
        batch = tensor.size(0)
        # TODO: Removing the clone().detach() will destroy the computation graph
        # Tests will fail
        return batch, tensor.clone().detach().view(batch, -1)

    def get_batch(self, module):
        return module.input0.size(0)

    def get_output(self, module):
        return module.output

    @staticmethod
    def _view_like(mat, like):
        """View as like with trailing and additional 0th dimension.

        If like is [N, C, H, ...], returns shape [-1, N, C, H, ...]
        """
        V = -1
        shape = (V, *like.shape)
        return try_view(mat, shape)

    @classmethod
    def view_like_input(cls, mat, module):
        return cls._view_like(mat, module.input0)

    @classmethod
    def view_like_output(cls, mat, module):
        return cls._view_like(mat, module.output)

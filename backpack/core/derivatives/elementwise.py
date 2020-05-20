"""Base class for more flexible Jacobians/Hessians of activation functions."""

from torch import einsum

from backpack.core.derivatives.basederivatives import BaseDerivatives


class ElementwiseDerivatives(BaseDerivatives):
    """Extended autodifferentiation functionality for element-wise activations.

    Element-wise functions have diagonal Jacobians/Hessians, since `output[i]`
    only depends on `input[i]`, and not on any `input[j ≠ i]`.

    The forward pass is `output[i] = f(input[i]) ∀ i`, where `f` denotes the
    activation function.

    Notes:
        - Methods that need to be implemented:
          - Required: `df`.
          - If the activation is piece-wise linear: `hessian_is_zero`, else `d2f`.
    """

    def df(self, module, g_inp, g_out):
        """Elementwise first derivative.

        Args:
            module (torch.nn.Module): PyTorch activation function module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.

        Returns:
            (torch.Tensor): Tensor containing the derivatives `f'(input[i]) ∀ i`.
        """

        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        """Elementwise second derivative.

        Only needs to be implemented for non piece-wise linear functions.

        Args:
            module (torch.nn.Module): PyTorch activation function module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.

        Returns:
            (torch.Tensor): Tensor containing the derivatives `f''(input[i]) ∀ i`.
        """

        raise NotImplementedError("Second derivatives not implemented")

    def hessian_diagonal(self, module, g_inp, g_out):
        """Return `∂²output[i] / ∂input[i]²`.

        Notes:
            - Only required if `hessian_is_diagonal` returns `True`.

        Args:
            module (torch.nn.Module): PyTorch activation function module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.
        """
        self._no_inplace(module)

        return self.d2f(module, g_inp, g_out) * g_out[0]

    def hessian_is_diagonal(self):
        """Elementwise activation function Hessians are diagonal.

        Returns:
            bool: True
        """
        return True

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        df_elementwise = self.df(module, g_inp, g_out)
        return einsum("...,v...->v...", (df_elementwise, mat))

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        batch, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum("ni,nj,ij->ij", (df_flat, df_flat, mat)) / batch

    def _make_residual_mat_prod(self, module, g_inp, g_out):
        def residual_mat_prod(mat):
            """Multiply with the residual: mat → [∑_{k} Hz_k(x) 𝛿z_k] mat.

            Second term of the module input Hessian backpropagation equation.
            """
            residual = self.d2f(module, g_inp, g_out) * g_out[0]
            return einsum("n...,vn...->vn...", (residual, mat))

        return residual_mat_prod

    @staticmethod
    def _no_inplace(module):
        """Do not support inplace modification.

        Jacobians/Hessians might be computed using the modified input instead
        of the original.

        Args:
            module (torch.nn.Module): Elementwise activation module.

        Raises:
            NotImplementedError: If `module` has inplace option enabled.

        Todo:
            - Write tests to investigate what happens with `inplace=True`.
        """
        has_inplace_option = hasattr(module, "inplace")

        if has_inplace_option:
            if module.inplace is True:
                raise NotImplementedError("Inplace not supported in {}.".format(module))

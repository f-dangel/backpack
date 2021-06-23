"""Derivatives of the identity module."""

from backpack.core.derivatives.basederivatives import BaseDerivatives


class IdentityDerivatives(BaseDerivatives):
    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """The (transposed) Jacobian is the identity."""
        return mat

"""
BackPACK implementation of a KFAC `CurvatureWrapper` for use with `FancyDamping`
"""

from backpack.gradient import backpack, extensions as ext

from .curvature_wrapper import CurvatureWrapper


class KFACCurvatureWrapper(CurvatureWrapper):
    def compute_derivatives_and_stuff(self, closure):
        #    def accumulate_derivatives_and_return_loss(self, closure):
        #        if self.should_compute_diag_ggn():
        #            with backpack(ext.DIAG_GGN):
        #                loss = closure()
        #        else:
        #            loss = closure()
        #        return loss
        raise NotImplementedError

    def __update_factors(self):
        # Update the estimates of the required A_i,j  and G_i,j
        # using the ai’s computed in forward pass for S1,
        # and the gi’s computed in the additional backwards pass for S1
        # (as described Section 5)
        raise NotImplementedError

    def reduction_ratio(self):
        pass

    def __compute_approx_fisher_inverse(self):
        # Compute the approximate Fisher inverse
        # (using the formulas derived in Section 4.2 or Section 4.3)
        # from versions of the current Ai,j ’s and Gi,j ’s
        # which are modified as per the factored Tikhonov damping technique
        # described in Section 6.3 (but using γ as described in Section 6.6)
        raise NotImplementedError

    def compute_step(self, inv_damping, trust_damping, l2_reg):
        step_proposal = self.__compute_step_proposal()
        corrected_step = self.__correct(step_proposal)
        return corrected_step

    def __compute_step_proposal(self):
        # Compute the update proposal ∆
        # by multiplying current estimate of approximate Fisher inverse
        # by the estimate of ∇h
        # (using the formulas derived in Section 4.2 or Section 4.3).
        # For layers with size d < m consider using trick
        # described at the end of Section 8 for increased efficiency.
        raise NotImplementedError

    def __correct(self, step):
        raise NotImplementedError
        # Compute the final update δ from ∆ as described in Section 6.4
        # (or Section 7 if using momentum)
        # where the matrix-vector products with F are estimated on S2
        # using the ai’s computed in the forward

    def inverse_candidate(self, inv_damping_candidate):
        raise NotImplementedError

    def invalidate_inverse_candidate(self):
        raise NotImplementedError

    def accept_inverse_candidate(self):
        raise NotImplementedError

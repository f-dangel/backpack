import warnings

import torch
from math import sqrt, inf as INFINITY
from torch.optim import Optimizer
from bpexts.gradient import bpexts, extensions as ext


def get_optimizer(model, lr):
    return DiagNGD(model.parameters(), lr=lr)


#    def accumulate_derivatives_and_return_loss(self, closure):
#        if self.should_compute_diag_ggn():
#            with bpexts(ext.DIAG_GGN):
#                loss = closure()
#        else:
#            loss = closure()
#        return loss


class CurvatureWrapper:
    def __init__(self):
        pass

    def compute_derivatives_and_stuff(self, closure):
        raise NotImplementedError("Should compute gradients and curvature updates")
        pass

    def reduction_ratio(self):
        """
        𝜌: reduction ratio
        𝜃: parameter
        𝛿: proposed update
        𝜌 = h(𝜃 + 𝛿) − h(𝜃)/M(𝛿)
        """
        raise NotImplementedError

    def compute_step_proposal(self):
        raise NotImplementedError

    def inverse_candidate(self, inv_damping_candidate):
        raise NotImplementedError

    def accept_inverse_candidate(self):
        raise NotImplementedError

    def invalidate_inverse_candidate(self):
        raise NotImplementedError


class KFACCurvatureWrapper(CurvatureWrapper):
    def damping_update(self):
        # the Levenberg-Marquardt style rule described in Section 6.5
        raise NotImplementedError

    def update_factors(self):
        # Update the estimates of the required A_i,j  and G_i,j
        # using the ai’s computed in forward pass for S1,
        # and the gi’s computed in the additional backwards pass for S1
        # (as described Section 5)
        raise NotImplementedError

    def compute_approx_fisher_inverse(self):
        # Compute the approximate Fisher inverse
        # (using the formulas derived in Section 4.2 or Section 4.3)
        # from versions of the current Ai,j ’s and Gi,j ’s
        # which are modified as per the factored Tikhonov damping technique
        # described in Section 6.3 (but using γ as described in Section 6.6)
        raise NotImplementedError

    def compute_update_proposal(self):
        # Compute the update proposal ∆
        # by multiplying current estimate of approximate Fisher inverse
        # by the estimate of ∇h
        # (using the formulas derived in Section 4.2 or Section 4.3).
        # For layers with size d < m consider using trick
        # described at the end of Section 8 for increased efficiency.
        raise NotImplementedError


class FancyDampingWrapper(Optimizer):

    ############################################################################
    # Init and Validation
    ############################################################################

    def __init__(self,
                 params,
                 curvature_wrapper: CurvatureWrapper,
                 l2_reg=0.,  # 𝜂
                 trust_damping=150.,  # 𝜆
                 trust_damping_factor=None,  # ω1
                 update_interval_trust_damping=5,  # T1
                 inv_damping=None,  # 𝛾
                 inv_damping_factor=None,  # ω2
                 update_interval_inv_damping=20,  # T2
                 update_interval_inversion=20,  # T3
                 ):

        self.step_counter = 0
        self.l2_reg = l2_reg
        self.trust_damping = trust_damping
        if inv_damping is None:
            self.inv_damping = sqrt(trust_damping + l2_reg)
        self.curvature_wrapper = curvature_wrapper

        MAGIC_FACTOR_FROM_KFAC_PAPER = 19 / 20
        MAGIC = MAGIC_FACTOR_FROM_KFAC_PAPER
        if inv_damping_factor is None:
            self.inv_damping_factor = sqrt(MAGIC) ** update_interval_inv_damping
        if trust_damping_factor is None:
            self.trust_damping_factor = MAGIC ** update_interval_trust_damping

        self.update_interval_trust_damping = update_interval_trust_damping
        self.update_interval_inv_damping = update_interval_inv_damping
        self.update_interval_inversion = update_interval_inversion
        self.update_intervals = [
            update_interval_trust_damping,
            update_interval_inv_damping,
            update_interval_inversion,
        ]

        self.validate_parameters()

        super().__init__(params)

    def validate_parameters(self):
        update_intervals_are_positive_ints = all(
            [isinstance(interval, int) for interval in self.update_intervals]
        ) and all(
            [interval > 0 for interval in self.update_intervals]
        )

        inv_damping_interval_is_multiple_of_inv_interval = (
                (self.update_interval_inv_damping %
                 self.update_interval_inversion) == 0
        )

        damping_factors_are_between_0_and_1 = (
                (0. < self.inv_damping_factor <= 1.) and
                (0. < self.trust_damping_factor <= 1.)
        )

        if not update_intervals_are_positive_ints:
            raise ValueError(
                "Update intervals need to be positive integers." +
                "Got [{}, {}, {}]".format(*self.update_intervals)
            )

        if not inv_damping_interval_is_multiple_of_inv_interval:
            raise ValueError(
                "Update interval for damping the inverse needs to be " +
                "a multiple of the interval for the update of the inverse. " +
                "Got {}, {}".format(
                    self.update_interval_inv_damping,
                    self.update_interval_inversion
                )
            )

        if not damping_factors_are_between_0_and_1:
            raise ValueError(
                "Damping factors need to be 0 < x <= 1. " +
                "Got {}, {}".format(
                    self.inv_damping_factor,
                    self.trust_damping_factor,
                )
            )

    ############################################################################
    # Main update
    ############################################################################

    def step(self, closure):

        loss = self.curvature_wrapper.compute_derivatives_and_stuff(closure)

        if self.should_update_inverse() or self.should_update_inv_damping():
            step = self.update_inverse_and_inv_damping_and_compute_step()
        else:
            step = self.just_compute_step()

        self.apply(step)
        self.step_counter += 1

        return loss

    def update_inverse_and_inv_damping_and_compute_step(self):

        best_candidate_score = INFINITY
        best_step = None
        inv_damping_candidates = self.damping.inv_damping_candidates()

        for inv_damping_candidate in inv_damping_candidates:

            if self.should_update_inverse():
                self.curvature_wrapper.inverse_candidate(inv_damping_candidate)

            step = self.just_compute_step()

            if len(inv_damping_candidates) == 1:
                self.curvature_wrapper.accept_inverse_candidate()
                return step
            else:
                candidate_score = self.evaluate_step(step)
                if candidate_score < best_candidate_score:
                    best_step = step
                    self.curvature_wrapper.accept_inverse_candidate()
                else:
                    self.curvature_wrapper.invalidate_inverse_candidate()

        return best_step

    def just_compute_step(self):
        step_proposal = self.curvature_wrapper.compute_step_proposal()
        corrected_step = self.correct(step_proposal)
        return corrected_step

    def correct(self, step_proposal):
        # Compute the final update δ from ∆ as described in Section 6.4
        # (or Section 7 if using momentum)
        # where the matrix-vector products with F are estimated on S2
        # using the ai’s computed in the forward
        raise NotImplementedError

    ############################################################################
    # Helpers
    ############################################################################

    def inv_damping_candidates(self):
        if self.should_update_inv_damping():
            return [
                self.inv_damping,
                self.inv_damping / self.inv_damping_factor,
                self.inv_damping * self.inv_damping_factor,
            ]
        else:
            return [self.inv_damping_factor]

    def should_update_inverse(self):
        return (
                self.set_counter < 3 or
                (self.step_counter % self.update_interval_inversion == 0)
        )

    def should_update_inv_damping(self):
        raise self.step_counter % self.update_interval_inv_damping == 0

    def should_update_trust_damping(self):
        raise self.step_counter % self.update_interval_trust_damping == 0

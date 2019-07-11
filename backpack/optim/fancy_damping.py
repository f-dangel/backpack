"""
Damping strategy developed for KFAC

Optimizing Neural Networks with Kronecker-factored Approximate Curvature
James Martens and Roger Grosse
https://arxiv.org/abs/1503.05671
"""

from math import sqrt, inf as INFINITY
from torch.optim.optimizer import Optimizer

from .curvature_wrapper import CurvatureWrapper

MAGIC_FACTOR_FROM_KFAC_PAPER = 19. / 20.
DEBUG = True


def debug(*str):
    if DEBUG:
        print(" " * 40, *str)


class FancyDampingWrapper(Optimizer):

    ############################################################################
    # Init and Validation
    ############################################################################

    def __init__(self,
                 params,
                 curvature_wrapper: CurvatureWrapper,
                 l2_reg=0.,  # eta
                 trust_damping=150.,  # lambda
                 inv_damping=None,  # gamma
                 trust_damping_factor=None,  # ω1
                 inv_damping_factor=None,  # ω2
                 update_interval_trust_damping=5,  # T1
                 update_interval_inv_damping=20,  # T2
                 update_interval_inversion=20,  # T3
                 ):
        """
        Implements the damping strategy developed for
        [KFAC](https://arxiv.org/abs/1503.05671)
        in a curvature-agnostic way;
        the approximation used for curvature is handled by a `CurvatureWrapper`.

        Args:
            params (iterable): iterable of parameters to optimize
            l2_reg (float, optional): L2 penalty (default: 0)
            trust_damping (float, optional): damping parameter for the
                trust-region check of model quality (default: 150.)
            inv_damping (float, optional): damping parameter for the
                inversion of curvature matrices
                (default: `sqrt(trust_damping, l2_reg`)
            trust_damping_factor (float, optional): multiplicative factor
                for changes in the trust damping factor
                (default: `(19/20)**update_interval_trust_damping`)
            inv_damping_factor (float, optional): multiplicative factor
                for changes in the inversion damping factor
                (default: `sqrt(19/20)**update_interval_inv_damping`)
            update_interval_trust_damping (int, optional): update trust_damping
                every _ steps (default: 5)
            update_interval_inv_damping (int, optional): update inv_damping
                every _ steps. Needs to be a multiple of
                `update_interval_inversion` (default: 20)
            update_interval_inversion (int, optional): update curvature inverse
                every _ steps (default: 20)


        """

        self.step_counter = 0

        self.curvature_wrapper = curvature_wrapper
        self.l2_reg = l2_reg
        self.trust_damping = trust_damping
        if inv_damping is None:
            self.inv_damping = sqrt(trust_damping + l2_reg)
        else:
            self.inv_damping = inv_damping

        MAGIC = MAGIC_FACTOR_FROM_KFAC_PAPER
        if inv_damping_factor is None:
            self.inv_damping_factor = sqrt(MAGIC) ** update_interval_inv_damping
        else:
            self.inv_damping_factor = inv_damping_factor

        if trust_damping_factor is None:
            self.trust_damping_factor = MAGIC ** update_interval_trust_damping
        else:
            self.trust_damping_factor = trust_damping_factor

        self.update_interval_trust_damping = update_interval_trust_damping
        self.update_interval_inv_damping = update_interval_inv_damping
        self.update_interval_inversion = update_interval_inversion

        super().__init__(params, {})
        self.__validate_parameters()

    def __validate_parameters(self):
        update_intervals_are_positive_ints = all([
            isinstance(self.update_interval_trust_damping, int),
            isinstance(self.update_interval_inv_damping, int),
            isinstance(self.update_interval_inversion, int),
        ]) and all([
            self.update_interval_trust_damping > 0,
            self.update_interval_inv_damping > 0,
            self.update_interval_inversion > 0,
        ])

        inv_damping_interval_is_multiple_of_inv_interval = (
                (self.update_interval_inv_damping %
                 self.update_interval_inversion) == 0
        )

        damping_factors_are_between_0_and_1 = (
                (0. < self.inv_damping_factor <= 1.) and
                (0. < self.trust_damping_factor <= 1.)
        )

        only_one_group_of_parameters = len(self.param_groups) == 1

        if not update_intervals_are_positive_ints:
            raise ValueError(
                "Update intervals need to be positive integers." +
                "Got [{}, {}, {}]".format(
                    self.update_interval_trust_damping,
                    self.update_interval_inv_damping,
                    self.update_interval_inversion,
                )
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

        if not only_one_group_of_parameters:
            raise ValueError(
                "Expected only one group of parameters. " +
                "Got {}".format(len(self.param_groups))
            )

    ############################################################################
    # Main update
    ############################################################################

    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
            and returns the loss.
        """

        loss = self.curvature_wrapper.compute_derivatives_and_stuff(closure)

        if self.__should_update_inverse() or self.__should_update_inv_damping():
            step = self.__update_inverse_and_inv_damping_and_compute_step()
        else:
            step = self.curvature_wrapper.compute_step(
                self.inv_damping, self.trust_damping, self.l2_reg
            )

        self.__apply_step(step)

        self.__update_trust_damping_if_needed()

        self.curvature_wrapper.end_of_step()
        self.step_counter += 1

        debug("inv_damping", self.inv_damping)
        debug("trust_damping", self.trust_damping)

        return loss

    def __update_inverse_and_inv_damping_and_compute_step(self):

        best_candidate_score = INFINITY
        best_step = None
        best_inv_damping = None
        inv_damping_candidates = self.__inv_damping_candidates()

        for inv_damping_candidate in inv_damping_candidates:

            if self.__should_update_inverse():
                self.curvature_wrapper.inverse_candidate(inv_damping_candidate)

            step = self.curvature_wrapper.compute_step(
                inv_damping_candidate, self.trust_damping, self.l2_reg
            )

            if len(inv_damping_candidates) == 1:
                self.curvature_wrapper.accept_inverse_candidate()
                return step
            else:
                candidate_score = self.curvature_wrapper.evaluate_step(step, self.trust_damping, self.l2_reg)

                if candidate_score < best_candidate_score:
                    best_step = step
                    best_inv_damping = inv_damping_candidate
                    best_candidate_score = candidate_score
                    self.curvature_wrapper.accept_inverse_candidate()
                else:
                    self.curvature_wrapper.invalidate_inverse_candidate()

        self.inv_damping = best_inv_damping
        return best_step

    ############################################################################
    # Helpers
    ############################################################################

    def __apply_step(self, step):
        group = self.param_groups[0]
        for p, dp in zip(group['params'], step):
            p.data.add_(dp)

    def __inv_damping_candidates(self):
        if self.__should_update_inv_damping():
            return [
                self.inv_damping,
                self.inv_damping / self.inv_damping_factor,
                self.inv_damping * self.inv_damping_factor,
            ]
        else:
            return [self.inv_damping_factor]

    def __should_update_inverse(self):
        return self.__should_update_inv_damping() or (
                self.step_counter < 3 or
                (self.step_counter % self.update_interval_inversion == 0)
        )

    def __should_update_inv_damping(self):
        return self.step_counter % self.update_interval_inv_damping == 0

    def __update_trust_damping_if_needed(self):
        should_update = (
                (self.step_counter % self.update_interval_trust_damping) == 0
        )

        if should_update:
            reduction_ratio = -self.curvature_wrapper.reduction_ratio(self.trust_damping, self.l2_reg)
            if reduction_ratio < .25:
                self.trust_damping /= self.trust_damping_factor
            elif reduction_ratio > .75:
                self.trust_damping *= self.trust_damping_factor

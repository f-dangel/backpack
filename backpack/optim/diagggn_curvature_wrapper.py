"""
BackPACK implementation of a Diag GGN `CurvatureWrapper`
for use with `FancyDamping`
"""
import warnings

import torch
from torch import zeros_like, ones_like

from backpack.hessian.free import R_op, hessian_vector_product
from backpack.gradient import backpack, extensions as ext
from .curvature_wrapper import CurvatureWrapper

MIN_DECAY = 0.95
EPS = 10**-6


class DiagGGNCurvatureWrapper(CurvatureWrapper):

    ############################################################################
    # Init and Validation
    ############################################################################

    def __init__(self, params):
        super().__init__(params)
        self.ggn_diags = list([ones_like(p) for p in self.parameters])
        self.ggn_diag_invs = list([ones_like(p) for p in self.parameters])
        self.ggn_diag_invs_tmp = []
        self.step_counter = 0

        self.last_loss = None
        self.last_output = None
        self.last_closure = None
        self.prev_step = list([zeros_like(p) for p in self.parameters])
        self.last_proposed_step = None

    ############################################################################
    # Main steps
    ############################################################################

    def compute_derivatives_and_stuff(self, closure):
        self.last_closure = closure
        with backpack(ext.DIAG_GGN):
            self.last_loss, self.last_output = closure()
            self.last_loss.backward(retain_graph=True)
            self.__update_factors()
        return self.last_loss

    def __update_factors(self):
        decay = min([1. - 1. / (self.step_counter + 1), MIN_DECAY])
        for p, p_diag in zip(self.parameters, self.ggn_diags):
            p_diag.mul_(decay).add_(1. - decay, p.diag_ggn)

    def inverse_candidate(self, inv_damping):
        self.invalidate_inverse_candidate()
        for diag in self.ggn_diags:
            self.ggn_diag_invs_tmp.append(1. / (diag.add(inv_damping**2)))

    def compute_step(self, inv_damping, trust_damping, l2_reg):
        step = []
        for p, diag_inv in zip(self.parameters, self.__current_diag_inv()):
            step.append(-p.grad * diag_inv)

        corrected_step = self.__correct(step, trust_damping, l2_reg)

        return corrected_step

    def __correct(self, step, trust_damping, l2_reg):

        J_prev = R_op(self.last_output, self.parameters, self.prev_step)
        J_curr = R_op(self.last_output, self.parameters, step)

        H_J_prev = hessian_vector_product(self.last_loss, self.last_output,
                                          J_prev)
        H_J_curr = hessian_vector_product(self.last_loss, self.last_output,
                                          J_prev)

        ip = self.__inner_product

        curr_G_curr = ip(J_curr, H_J_curr)
        curr_G_prev = ip(J_curr, H_J_prev)
        prev_G_prev = ip(J_prev, H_J_prev)

        damp = trust_damping + l2_reg
        curr_prev = ip(step, self.prev_step) * damp
        curr_curr = ip(step, step) * damp
        prev_prev = ip(self.prev_step, self.prev_step) * damp

        M = torch.tensor([
            [curr_G_curr + curr_curr + EPS, curr_G_prev + curr_prev],
            [curr_G_prev + curr_prev, prev_G_prev + prev_prev + EPS],
        ])
        v = torch.tensor([
            [ip(step, [p.grad for p in self.parameters])],
            [ip(self.prev_step, [p.grad for p in self.parameters])],
        ])
        factors, _ = torch.solve(v, M)
        del _

        corrected_step = []
        for curr, prev in zip(step, self.prev_step):
            corrected_step.append(-(factors[0] * curr + factors[1] * prev))

        self.last_proposed_step = corrected_step
        return corrected_step

    def reduction_ratio(self, trust_damping, l2_reg):
        loss_after_step = self.last_closure()[0]
        loss_before_step = self.last_loss

        loss_change = loss_after_step - loss_before_step

        predicted_new_loss = self.__model_predict(self.last_proposed_step,
                                                  trust_damping, l2_reg)
        rho = loss_change / predicted_new_loss

        #        print("    loss before step:   ", loss_before_step)
        #        print("    loss after step:    ", loss_after_step)
        #        print("    predicted new loss: ", predicted_new_loss)
        return ((loss_after_step - loss_before_step) / predicted_new_loss)

    def __model_predict(self, step, trust_damping, l2_reg):
        J_v = R_op(self.last_output, self.parameters, step)
        H_J_v = hessian_vector_product(self.last_loss, self.last_output, J_v)

        ip = self.__inner_product
        return (.5 * ip(J_v, H_J_v) +
                .5 * (trust_damping + l2_reg) * ip(step, step) + ip(
                    step, [p.grad for p in self.parameters]))

    def evaluate_step(self, step, trust_damping, l2_reg):
        return self.__model_predict(step, trust_damping, l2_reg)

    ############################################################################
    # Helpers and management
    ############################################################################

    def end_of_step(self):
        self.step_counter += 1

        self.prev_step = self.last_proposed_step
        del self.last_proposed_step

        self.invalidate_inverse_candidate()

    def invalidate_inverse_candidate(self):
        while len(self.ggn_diag_invs_tmp) > 0:
            del self.ggn_diag_invs_tmp[0]

    def accept_inverse_candidate(self):
        while len(self.ggn_diag_invs) > 0:
            del self.ggn_diag_invs[0]
        for ggn_diag in self.ggn_diag_invs_tmp:
            self.ggn_diag_invs.append(ggn_diag)
        self.invalidate_inverse_candidate()

    def __current_diag_inv(self):
        if len(self.ggn_diag_invs_tmp) > 0:
            return self.ggn_diag_invs_tmp
        else:
            return self.ggn_diag_invs

    def __inner_product(self, xs, ys):
        res = 0
        for x, y in zip(xs, ys):
            res += torch.sum(x * y)
        return res

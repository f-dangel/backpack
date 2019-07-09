"""
BackPACK implementation of a Diag GGN `CurvatureWrapper`
for use with `FancyDamping`
"""
import warnings

from torch import zeros_like

from bpexts.gradient import bpexts, extensions as ext
from .curvature_wrapper import CurvatureWrapper

MIN_DECAY = 0.95


class DiagGGNCurvatureWrapper(CurvatureWrapper):

    def __init__(self, params):
        super().__init__(params)
        self.ggn_diags = list([
            zeros_like(p) for p in self.__params()
        ])
        self.ggn_diag_invs = list([
            zeros_like(p) for p in self.__params()
        ])
        self.ggn_diag_invs_tmp = []
        self.step_counter = 0

    def compute_derivatives_and_stuff(self, closure):
        with bpexts(ext.DIAG_GGN):
            loss = closure()
            loss.backward()
            self.__update_factors()
        return loss

    def __update_factors(self):
        decay = min([1. - 1. / (self.step_counter + 1), MIN_DECAY])
        for p, p_diag in zip(self.__params(), self.ggn_diags):
            p_diag.mul_(decay).add_(1. - decay, p.diag_ggn)

    def inverse_candidate(self, inv_damping):
        if not len(self.ggn_diag_invs_tmp) == 0:
            self.ggn_diag_invs_tmp = []
        for diag in self.ggn_diags:
            self.ggn_diag_invs_tmp.append(1. / (diag.add(inv_damping)))

    def invalidate_inverse_candidate(self):
        while len(self.ggn_diag_invs_tmp) > 0:
            del self.ggn_diag_invs_tmp[0]

    def accept_inverse_candidate(self):
        while len(self.ggn_diag_invs) > 0:
            del self.ggn_diag_invs[0]
        for ggn_diag in self.ggn_diag_invs_tmp:
            self.ggn_diag_invs.append(ggn_diag)
        self.invalidate_inverse_candidate()

    def compute_step(self, inv_damping, trust_damping):
        step_proposal = self.__compute_step_proposal()
        corrected_step = self.__correct(step_proposal)
        return corrected_step

    def __params(self):
        return self.parameters

    def __current_diag_inv(self):
        if len(self.ggn_diag_invs_tmp) > 0:
            return self.ggn_diag_invs_tmp
        else:
            return self.ggn_diag_invs

    def __compute_step_proposal(self):
        step = []
        for p, diag_inv in zip(self.__params(), self.__current_diag_inv()):
            step.append(-p/diag_inv)
        return step

    def __correct(self, step):
        warnings.warn("correct does nothing")
        # Compute the final update δ from ∆ as described in Section 6.4
        # (or Section 7 if using momentum)
        # where the matrix-vector products with F are estimated on S2
        # using the ai’s computed in the forward
        return step

    def reduction_ratio(self):
        warnings.warn("correct does nothing")
        return .5

    def evaluate_step(self, step):
        warnings.warn("correct does nothing")
        return 1.

    def end_of_step(self):
        self.step_counter += 1
        self.invalidate_inverse_candidate()

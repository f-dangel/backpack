import torch
from .implementation import Implementation
import bpexts.hessian.free as HF


class AutogradImpl(Implementation):

    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        batch_grads = [
            torch.zeros(self.N, *p.size()).to(self.device)
            for p in self.model.parameters()
        ]

        for b in range(self.N):
            gradients = torch.autograd.grad(self.loss(b), self.model.parameters())

            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() / self.N

        return batch_grads

    def batch_l2(self):
        batch_grad = self.batch_gradients()
        batch_l2 = [(g**2).sum(list(range(1, len(g.shape)))) for g in batch_grad]
        return batch_l2

    def variance(self):
        batch_grad = self.batch_gradients()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

    def sgs(self):
        batch_grad = self.batch_gradients()
        sgs = [(g**2).sum(0) for g in batch_grad]
        return sgs

    def diag_ggn(self):
        outputs = self.model(self.problem.X)
        loss = self.problem.lossfunc(outputs, self.problem.Y)

        tot_params = sum([p.numel() for p in self.model.parameters()])

        def extract_ith_element_of_diag_ggn(i):
            v = torch.zeros(tot_params).to(self.device)
            v[i] = 1.

            vs = HF.vector_to_parameter_list(
                v, list(self.model.parameters()))

            GGN_vs = HF.ggn_vector_product(loss, outputs, self.model, vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diagonal_index = 0
        diag_ggns = []
        for p in list(self.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(diagonal_index)
                diag_ggn_p[parameter_index] = diag_value
                diagonal_index += 1

            diag_ggns.append(diag_ggn_p.view(p.size()))

        return diag_ggns

    def diag_h(self):
        loss = self.problem.lossfunc(self.model(self.problem.X), self.problem.Y)
        tot_params = sum([p.numel() for p in self.model.parameters()])

        def extract_ith_element_of_diag_h(i):
            v = torch.zeros(tot_params).to(self.device)
            v[i] = 1.
            plist = list(self.model.parameters())
            vs = HF.vector_to_parameter_list(v, plist)
            Hvs = HF.hessian_vector_product(loss, plist, vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])
            return Hv[i]

        diagonal_index = 0
        diag_hs = []
        for p in list(self.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(diagonal_index)
                diag_h_p[parameter_index] = diag_value
                diagonal_index += 1

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

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
        sgs = self.plist_like(self.model.parameters())

        for b in range(self.N):
            gradients = torch.autograd.grad(self.loss(b), self.model.parameters())
            for idx, g in enumerate(gradients):
                sgs[idx] += (g.detach() / self.N) ** 2

        return sgs

    def diag_ggn(self):
        outputs = self.model(self.problem.X)
        loss = self.problem.lossfunc(outputs, self.problem.Y)

        def extract_ith_element_of_diag_ggn(i, p):
            v = torch.zeros(p.numel()).to(self.device)
            v[i] = 1.
            vs = HF.vector_to_parameter_list(v, [p])
            GGN_vs = HF.ggn_vector_product_from_plist(loss, outputs, [p], vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diag_ggns = []
        for p in list(self.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))

        return diag_ggns

    def diag_h(self):
        loss = self.problem.lossfunc(self.model(self.problem.X), self.problem.Y)

        def extract_ith_element_of_diag_h(i, p):
            v = torch.zeros(p.numel()).to(self.device)
            v[i] = 1.
            vs = HF.vector_to_parameter_list(v, [p])
            Hvs = HF.hessian_vector_product(loss, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])
            return Hv[i]

        diag_hs = []
        for p in list(self.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

    def plist_like(plist):
        return list([
            torch.zeros(*p.size()).to(self.device)
            for p in plist
        ])

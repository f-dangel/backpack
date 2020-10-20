from test.extensions.implementation.base import ExtensionsImplementation
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.hessianfree.rop import R_op

import torch


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def batch_grad(self):
        N = self.problem.input.shape[0]
        batch_grads = [
            torch.zeros(N, *p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            gradients_list.append(gradients)
            loss_list[b] = loss

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for b, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() * factor

        return batch_grads

    def batch_l2_grad(self):
        batch_grad = self.batch_grad()
        batch_l2_grads = [(g ** 2).flatten(start_dim=1).sum(1) for g in batch_grad]
        return batch_l2_grads

    def sgs(self):
        N = self.problem.input.shape[0]
        sgs = [
            torch.zeros(*p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            loss_list[b] = loss
            gradients_list.append(gradients)

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for _, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                sgs[idx] += (g.detach() * factor) ** 2
        return sgs

    def variance(self):
        batch_grad = self.batch_grad()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

    def diag_ggn(self):
        _, output, loss = self.problem.forward_pass()

        def extract_ith_element_of_diag_ggn(i, p):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])
            GGN_vs = ggn_vector_product_from_plist(loss, output, [p], vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diag_ggns = []
        for p in list(self.problem.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))
        return diag_ggns

    def diag_h(self):
        _, _, loss = self.problem.forward_pass()

        def hvp(df_dx, x, v):
            Hv = R_op(df_dx, x, v)
            return [j.detach() for j in Hv]

        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])

            Hvs = hvp(df_dx, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.problem.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            df_dx = torch.autograd.grad(loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

import torch
from .implementation import Implementation

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.lop import L_op
from backpack.hessianfree.rop import R_op
from backpack.hessianfree.utils import vector_to_parameter_list


class AutogradImpl(Implementation):
    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        batch_grads = [
            torch.zeros(self.N, *p.size()).to(self.device)
            for p in self.model.parameters()
        ]

        for b in range(self.N):
            gradients = torch.autograd.grad(
                self.loss(b), self.model.parameters())
            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() / self.N

        return batch_grads

    def batch_l2(self):
        batch_grad = self.batch_gradients()
        batch_l2 = [(g**2).sum(list(range(1, len(g.shape))))
                    for g in batch_grad]
        return batch_l2

    def variance(self):
        batch_grad = self.batch_gradients()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

    def sgs(self):
        sgs = self.plist_like(self.model.parameters())

        for b in range(self.N):
            gradients = torch.autograd.grad(
                self.loss(b), self.model.parameters())
            for idx, g in enumerate(gradients):
                sgs[idx] += (g.detach() / self.N)**2

        return sgs

    def diag_ggn(self):
        outputs = self.model(self.problem.X)
        loss = self.problem.lossfunc(outputs, self.problem.Y)

        def extract_ith_element_of_diag_ggn(i, p):
            v = torch.zeros(p.numel()).to(self.device)
            v[i] = 1.
            vs = vector_to_parameter_list(v, [p])
            GGN_vs = ggn_vector_product_from_plist(loss, outputs, [p], vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diag_ggns = []
        for p in list(self.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(
                    parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))

        return diag_ggns

    def diag_h(self):
        loss = self.problem.lossfunc(
            self.model(self.problem.X), self.problem.Y)

        def hvp(df_dx, x, v):
            Hv = R_op(df_dx, x, v)
            return tuple([j.detach() for j in Hv])

        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = torch.zeros(p.numel()).to(self.device)
            v[i] = 1.
            vs = vector_to_parameter_list(v, [p])

            Hvs = hvp(df_dx, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            df_dx = torch.autograd.grad(
                loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(
                    parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

    def h_blocks(self):
        mat_list = []
        for p in self.model.parameters():
            mat_list.append(torch.eye(p.numel(), device=p.device))
        return self.hmp(mat_list)

    def hvp(self, vec_list):
        mat_list = [vec.unsqueeze(-1) for vec in vec_list]
        results = self.hmp(mat_list)
        results_vec = [mat.squeeze(-1) for mat in results]
        return results_vec

    def hmp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))

        loss = self.problem.lossfunc(
            self.model(self.problem.X), self.problem.Y)

        results = []
        for p, mat in zip(self.model.parameters(), mat_list):
            results.append(self.hvp_applied_columnwise(loss, p, mat))

        return results

    def hvp_applied_columnwise(self, f, p, mat):
        h_cols = []
        for i in range(mat.size(1)):
            hvp_col_i = hessian_vector_product(f, [p], mat[:, i].view_as(p))[0]
            h_cols.append(hvp_col_i.view(-1, 1))

        return torch.cat(h_cols, dim=1)

    def ggn_blocks(self):
        mat_list = []
        for p in self.model.parameters():
            mat_list.append(torch.eye(p.numel(), device=p.device))
        return self.ggn_mp(mat_list)

    def ggn_vp(self, vec_list):
        mat_list = [vec.unsqueeze(-1) for vec in vec_list]
        results = self.ggn_mp(mat_list)
        results_vec = [mat.squeeze(-1) for mat in results]
        return results_vec

    def ggn_mp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))

        outputs = self.model(self.problem.X)
        loss = self.problem.lossfunc(outputs, self.problem.Y)

        results = []
        for p, mat in zip(self.model.parameters(), mat_list):
            results.append(
                self.ggn_vp_applied_columnwise(loss, outputs, p, mat))

        return results

    def ggn_vp_applied_columnwise(self, loss, out, p, mat):
        ggn_cols = []
        for i in range(mat.size(1)):
            col_i = vector_to_parameter_list(mat[:, i], [p])

            GGN_col_i = ggn_vector_product_from_plist(loss, out, [p], col_i)
            GGN_col_i = torch.cat([g.detach().view(-1) for g in GGN_col_i])
            ggn_cols.append(GGN_col_i.view(-1, 1))

        return torch.cat(ggn_cols, dim=1)

    def plist_like(self, plist):
        return list([torch.zeros(*p.size()).to(self.device) for p in plist])

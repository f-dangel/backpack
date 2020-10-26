from test.extensions.implementation.base import ExtensionsImplementation

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

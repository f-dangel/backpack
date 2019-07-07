import torch
from torch.optim import SGD, Optimizer
from torch.optim.optimizer import required

MOMENTUM = 0.9


class DiagNGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, damping=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum, damping=damping)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p)
                    d_p = param_state['momentum_buffer']

                p.data.add_(-group['lr'], d_p / (p.diag_ggn.data + group["damping"]))

        return loss


def get_optimizer(model, lr):
    return DiagNGD(model.parameters(), lr=lr, momentum=MOMENTUM)

class Implementation:
    def __init__(self, test_problem, device=None):
        self.problem = test_problem
        self.model = self.problem.model
        self.N = self.problem.N
        if device is not None:
            self.problem.to(device)
            self.device = device
        else:
            self.device = self.problem.device

    def to(self, device):
        self.model.to(device)
        return self

    def loss(self, b=None):
        return self.problem.loss(b)

    def clear(self):
        self.problem.clear()

    def gradient(self):
        raise NotImplementedError

    def batch_gradients(self):
        raise NotImplementedError

    def batch_l2(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def sgs(self):
        raise NotImplementedError

    def diag_ggn(self):
        raise NotImplementedError

    def diag_h(self):
        raise NotImplementedError

    def hmp(self, mat_list):
        raise NotImplementedError

class ExtensionsImplementation:
    """Base class for autograd and BackPACK implementations of extensions."""

    def __init__(self, problem):
        self.problem = problem

    def batch_grad(self):
        """Individual gradients."""
        raise NotImplementedError

    def batch_l2_grad(self):
        """L2 norm of Individual gradients."""
        raise NotImplementedError

    def sgs(self):
        """Sum of Square of Individual gradients"""
        raise NotImplementedError

    def variance(self):
        """Variance of Individual gradients"""
        raise NotImplementedError

    def diag_ggn(self):
        """Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_ggn_mc(self, mc_samples):
        """MC approximation of Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_h(self):
        """Diagonal of Hessian"""
        raise NotImplementedError

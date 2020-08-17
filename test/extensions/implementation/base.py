class ExtensionsImplementation:
    """Base class for autograd and BackPACK implementations of extensions."""

    def __init__(self, problem):
        self.problem = problem

    def batch_grad(self):
        """Individual gradients."""
        raise NotImplementedError

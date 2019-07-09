"""
Interface for integrating a curvature approximation with fancy_damping
"""


class CurvatureWrapper:
    def __init__(self, parameters):
        self.parameters = parameters

    def compute_derivatives_and_stuff(self, closure):
        raise NotImplementedError("Should compute gradients and curvature updates")

    def compute_step(self, inv_damping, trust_damping):
        raise NotImplementedError

    def evaluate_step(self, step):
        raise NotImplementedError

    def reduction_ratio(self):
        """
        Compare the improvement in loss due to the step against
        the expected improvement under the quadratic model.

        If `𝜃` is the starting parameter, `𝛿` is the step taken,
        `L` is the loss function and `M` the quadratic model,
        the reduction ratio is given by
        ```
        h(𝜃 + 𝛿) − h(𝜃)/M(𝛿)
        ```
        """
        raise NotImplementedError

    def inverse_candidate(self, inv_damping_candidate):
        """
        Compute a new inverse candidate using the given damping,
        to be used on future calls to `compute_step`.

        It should be possible to go back to the previously valid inverse,
        by calling `invalidate_inverse_candidate`.

        The new inverse candidate will be validated,
        and the previous one forgotten,
        by calling `accept_inverse_candidate`.
        """
        raise NotImplementedError

    def accept_inverse_candidate(self):
        """See `inverse_candidate`"""
        raise NotImplementedError

    def invalidate_inverse_candidate(self):
        """See `inverse_candidate`"""
        raise NotImplementedError

    def end_of_step(self):
        raise NotImplementedError

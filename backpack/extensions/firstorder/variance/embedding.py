"""Variance extension for Embedding."""
from backpack.extensions.firstorder.gradient.embedding import GradEmbedding
from backpack.extensions.firstorder.sum_grad_squared.embedding import SGSEmbedding
from backpack.extensions.firstorder.variance.variance_base import VarianceBaseModule


class VarianceEmbedding(VarianceBaseModule):
    """Variance extension for Embedding."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            grad_extension=GradEmbedding(),
            sgs_extension=SGSEmbedding(),
            params=["weight"],
        )

"""Gradient extension for Embedding."""
from backpack.core.derivatives.embedding import EmbeddingDerivatives
from backpack.extensions.firstorder.gradient.base import GradBaseModule


class GradEmbedding(GradBaseModule):
    """Gradient extension for Embedding."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=EmbeddingDerivatives(), params=["weight"])

"""SGS extension for Embedding."""
from backpack.core.derivatives.embedding import EmbeddingDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSEmbedding(SGSBase):
    """SGS extension for Embedding."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=EmbeddingDerivatives(), params=["weight"])

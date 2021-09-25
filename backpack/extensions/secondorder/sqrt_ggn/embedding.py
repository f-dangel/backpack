"""Contains extension for the embedding layer used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.embedding import EmbeddingDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNEmbedding(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Embedding`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Embedding`` module."""
        super().__init__(EmbeddingDerivatives(), params=["weight"])

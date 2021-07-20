"""Contains base classes for second order extensions."""
from backpack import BackpropExtension


class SecondOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for second order."""

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return True

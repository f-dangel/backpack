"""Class for saving backpropagation quantities."""
from typing import Dict, Union

from torch import Tensor


class SavedQuantities:
    """Implements interface to save backpropagation quantities."""

    def __init__(self):
        """Initialization."""
        self._saved_quantities: Dict[int, Tensor] = {}

    def save_quantity(self, key: int, quantity: Tensor) -> None:
        """Saves the quantity under the specified key.

        Args:
            key: id of grad_input[0]
            quantity: tensor to save

        Raises:
            NotImplementedError: if the key already exists
        """
        if key in self._saved_quantities:
            # TODO if exists: accumulate quantities (ResNet)
            raise NotImplementedError(
                "Quantity with given key already exists. Multiple backpropagated "
                "quantities like in ResNets are not supported yet."
            )
        else:
            self._saved_quantities[key] = quantity

    def retrieve_quantity(self, key: int) -> Union[Tensor, None]:
        """Returns the saved quantity.

        Args:
            key: id of grad_output[0]

        Returns:
            the saved quantity, None if it does not exist
        """
        return self._saved_quantities.pop(key, None)

    def clear(self) -> None:
        """Clear saved quantities."""
        self._saved_quantities.clear()

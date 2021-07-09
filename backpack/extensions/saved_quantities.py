"""Class for saving backpropagation quantities."""
from typing import Dict, Union

from torch import Tensor


class SavedQuantities:
    """Implements interface to save backpropagation quantities."""

    def __init__(self):
        """Initialization."""
        self._saved_quantities: Dict[int, Tensor] = {}

    def save_quantity(self, key: int, quantity: Tensor) -> bool:
        """Saves the quantity under the specified key.

        Args:
            key: id of grad_input[0]
            quantity: tensor to save

        Returns:
            whether save was successful
        """
        if key in self._saved_quantities:
            self._saved_quantities[key] = quantity
            # TODO clear saved quantities after backward finished
            # this avoids that the quantity exists from previous backward

            # TODO if exists: accumulate quantities (ResNet)
        else:
            self._saved_quantities[key] = quantity
            return True

    def retrieve_quantity(self, key: int) -> Union[Tensor, None]:
        """Returns the saved quantity.

        Args:
            key: id of grad_output[0]

        Returns:
            the saved quantity, None if it does not exist
        """
        return self._saved_quantities.pop(key, None)

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
            key: data_ptr() of reference tensor (module.input0).
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

    def retrieve_quantity(self, key: int, delete_old: bool) -> Union[Tensor, None]:
        """Returns the saved quantity.

        Args:
            key: data_ptr() of reference tensor.
                For torch>=1.9.0 the reference tensor is grad_output[0].
                For older versions the reference tensor is module.output.
            delete_old: whether to delete the old quantity

        Returns:
            the saved quantity, None if it does not exist
        """
        get_value = (
            self._saved_quantities.pop if delete_old else self._saved_quantities.get
        )
        return get_value(key, None)

"""Class for saving backpropagation quantities."""
from typing import Any, Callable, Dict, Union

from torch import Tensor


class SavedQuantities:
    """Implements interface to save backpropagation quantities."""

    def __init__(self):
        """Initialization."""
        self._saved_quantities: Dict[int, Tensor] = {}

    def save_quantity(
        self,
        key: int,
        quantity: Tensor,
        accumulation_function: Callable[[Any, Any], Any],
    ) -> None:
        """Saves the quantity under the specified key.

        If the there is already an entry under the given key,
        the quantities are accumulated.

        Args:
            key: data_ptr() of reference tensor (module.input0).
            quantity: tensor to save
            accumulation_function: function defining how to accumulate quantity
        """
        if key in self._saved_quantities:
            # if exists: accumulate quantities (ResNet)
            self._saved_quantities[key] = accumulation_function(
                self.retrieve_quantity(key, delete_old=True), quantity
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

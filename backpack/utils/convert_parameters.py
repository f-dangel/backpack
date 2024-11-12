"""Utility functions to convert between parameter lists and vectors."""

from typing import Iterable, List

from torch import Tensor, typename


def vector_to_parameter_list(vec: Tensor, parameters: Iterable[Tensor]) -> List[Tensor]:
    """Convert the vector `vec` to a parameter-list format matching `parameters`.

    This function is the inverse of `parameters_to_vector` from the
    pytorch module `torch.nn.utils.convert_parameters`.
    Contrary to `vector_to_parameters`, which replaces the value
    of the parameters, this function leaves the parameters unchanged and
    returns a list of parameter views of the vector.

    ```
    from torch.nn.utils import parameters_to_vector

    vector_view = parameters_to_vector(parameters)
    param_list_view = vector_to_parameter_list(vec, parameters)

    for a, b in zip(parameters, param_list_view):
        assert torch.all_close(a, b)
    ```

    Args:
        vec: A single vector represents the parameters of a model
        parameters: An iterator of Tensors that are of the desired shapes.

    Raises:
        TypeError: If `vec` is not a PyTorch tensor.

    Returns:
        List of parameter-shaped tensors containing the entries of `vec`.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, Tensor):
        raise TypeError(f"expected Tensor, but got: {typename(vec)}")

    params_new = []
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it
        param_new = vec[pointer : pointer + num_param].view_as(param).data
        params_new.append(param_new)
        # Increment the pointer
        pointer += num_param

    return params_new

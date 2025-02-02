"""Helper functions for tests."""

from typing import Any, List


def chunk_sizes(total_size: int, num_chunks: int) -> List[int]:
    """Return list containing the sizes of chunks.

    Args:
        total_size: Total computation work.
        num_chunks: Maximum number of chunks the work will be split into.

    Returns:
        List of chunks with split work.
    """
    chunk_size = max(total_size // num_chunks, 1)

    if chunk_size == 1:
        sizes = total_size * [chunk_size]
    else:
        equal, rest = divmod(total_size, chunk_size)
        sizes = equal * [chunk_size]

        if rest != 0:
            sizes.append(rest)

    return sizes


def popattr(obj: Any, name: str) -> Any:
    """Pop an attribute from an object.

    Args:
        obj: The object from which to pop the attribute.
        name: The name of the attribute to pop.

    Returns:
        The attribute's value.
    """
    value = getattr(obj, name)
    delattr(obj, name)
    return value

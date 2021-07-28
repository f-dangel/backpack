"""Helper functions for tests."""

from typing import List


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

"""Utility functions for the test suite."""

from pytorch_memlab import MemReporter


def pytorch_current_memory_usage():
    """Return current memory usage in PyTorch (all devices)."""
    reporter = MemReporter()
    reporter.collect_tensor()
    reporter.get_stats()

    total_mem = 0
    for _, tensor_stats in reporter.device_tensor_stat.items():
        for stat in tensor_stats:
            _, _, _, mem = stat
            total_mem += mem

    return total_mem

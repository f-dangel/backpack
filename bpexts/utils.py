"""Utility functions."""

import gc
import numpy
import torch
import random


def torch_allclose(a, b, *args, **kwargs):
    """Return if two tensors are element-wise equal within a tolerance.

    TODO: Replace calls by :meth:`torch.allclose`.
    """
    return numpy.allclose(a.data, b.data, *args, **kwargs)


def torch_contains_nan(tensor):
    """Return whether a tensor contains NaNs.

    Parameters
    ----------
    tensor : :obj:`torch.Tensor`
        Tensor to be checked for NaNs.

    Returns
    -------
    bool
        If at least one NaN is contained in :obj:`tensor`.
    """
    return any(tensor.view(-1) != tensor.view(-1))


def set_seeds(seed=None):
    """Set random seeds of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy`,
    :mod:`random`.

    Per default, no reset will be performed.

    Parameters
    ----------
    seed : :obj:`int` or :obj:`None`, optional
        Seed initialization value, no reset if unspecified
    """
    if seed is not None:
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # NumPy
        numpy.random.seed(seed)
        # random
        random.seed(seed)


def memory_report():
    """Report the memory usage of the :obj:`torch.tensor.storage` both
    on CPUs and GPUs.

    Returns
    -------
    tuple
        Two tuples, each consisting of the number of allocated tensor
        elements and the total storage in MB on GPU and CPU, respectively.

    Notes
    -----
    * The code is a modified version from the snippet provided by
      https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
    """

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type. Ignore sparse tensors.

        There are two major storage types in major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory

        Parameters
        ----------
        tensors : (list(torch.Tensor))
            The tensors of specified type
        mem_type : (str)
            'CPU' or 'GPU' in current implementation

        Returns
        -------
        total_numel, total_mem : (int, float)
            Total number of allocated elements and total memory reserved
        """
        print('Storage on {}\n{}'.format(mem_type, '-' * LEN))
        total_numel, total_mem, visited_data = 0, 0., []

        # sort by size
        sorted_tensors = sorted(tensors, key=lambda t: t.storage().data_ptr())

        for tensor in sorted_tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()

            numel = tensor.storage().size()
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024.**2  # 32bit = 4Byte, MByte
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('{}  \t{}\t\t{:.7f}\t\t{}'.format(element_type, size, mem,
                                                    data_ptr))

            if data_ptr not in visited_data:
                total_numel += numel
                total_mem += mem
            visited_data.append(data_ptr)

        print('{}\nTotal Tensors (not counting shared multiple times):'
              '{}\nUsed Memory Space: {:.7f} MB\n{}'.format(
                  '-' * LEN, total_numel, total_mem, '-' * LEN))
        return total_numel, total_mem

    gc.collect()
    LEN = 65
    print('=' * LEN)
    objects = gc.get_objects()
    print('{}\t{}\t\t\t{}'.format('Element type', 'Size', 'Used MEM(MB)'))

    tensors = []
    for obj in objects:
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data')
                                        and torch.is_tensor(obj.data)):
                tensors.append(obj)
        except Exception:
            pass

    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    gpu_stats = _mem_report(cuda_tensors, 'GPU')
    cpu_stats = _mem_report(host_tensors, 'CPU')
    print('=' * LEN)

    return gpu_stats, cpu_stats

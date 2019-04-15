"""Utility functions."""

import gc
import numpy
import torch
import random
import math


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


def boxed_message(message):
    """Draw a box around a message.

    Parameters:
    -----------
    message : str
        Message to be boxed

    Returns:
    --------
    str
        Boxed message

    References:
    -----------
    - https://github.com/quapka/expecto-anything/blob/master/boxed_msg.py
    """

    def format_line(line, max_length):
        half_diff = (max_length - len(line)) / 2
        return "{}{}{}{}{}".format('| ', ' ' * math.ceil(half_diff), line,
                                   ' ' * math.floor(half_diff), ' |\n')

    lines = message.split('\n')
    max_length = max([len(l) for l in lines])
    horizontal = "{}{}{}".format('+', '-' * (max_length + 2), '+\n')
    result = horizontal
    for l in lines:
        result += format_line(l, max_length)
    result += horizontal
    return result.strip()


def same_padding2d(input_dim, kernel_dim, stride_dim):
    """Determine 2d padding parameters for same output size.

    Implementation modified from A. Bahde.

    Parameters:
    -----------
    input_dim : tuple(int)
        Width and height of the input image
    kernel_dim : tuple(int) or int
        Width and height of the kernel filter, assume quadratic
        filter size if ``int``.
    stride_dim : tuple(int) or int
        Stride dimensions, assume same in both directions if ``int``.

    Returns:
    --------
    tuple(int)
        Padding for left, right, top, and bottom margin.
    """
    in_height, in_width = input_dim
    kernel_height, kernel_width = 2 * (kernel_dim, ) if isinstance(
        kernel_dim, int) else kernel_dim
    stride_height, stride_width = 2 * (stride_dim, ) if isinstance(
        stride_dim, int) else stride_dim

    # output size to be achieved by padding
    out_height = math.ceil(in_height / stride_height)
    out_width = math.ceil(in_width / stride_width)

    # pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0)
    pad_along_width = max(
        (out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


def same_padding2d_before_forward(Layer):
    """Set padding parameters before forward pass."""

    class LayerSamePadding(Layer):
        def forward(self, x):
            assert self.stride == 1 or self.stride == (1, 1)
            pad_left, pad_right, pad_top, pad_bottom = same_padding2d(
                input_dim=(x.size(2), x.size(3)),
                kernel_dim=self.kernel_size,
                stride_dim=self.stride)
            if pad_left != pad_right or pad_top != pad_bottom:
                raise ValueError('Asymmetric padding not suported.')
            self.padding = (pad_left, pad_bottom)
            out = super().forward(x)
            if not x.size()[2:] == out.size()[2:]:
                raise ValueError("Expect same sizes, but got {}, {}".format(
                    x.size(), out.size()))
            return out

    return LayerSamePadding


class Conv2dSame(same_padding2d_before_forward(torch.nn.Conv2d)):
    """2d Convolution with padding same."""
    pass

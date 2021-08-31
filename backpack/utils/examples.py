"""Utility functions for examples."""
from typing import Iterator, List, Tuple

from torch import Tensor, stack, zeros
from torch.nn import Module
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list


def load_mnist_dataset() -> Dataset:
    """Download and normalize MNIST training data.

    Returns:
        Normalized MNIST dataset
    """
    return MNIST(
        root="./data",
        train=True,
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        download=True,
    )


def get_mnist_dataloader(batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Returns a dataloader for MNIST.

    Args:
        batch_size: Mini-batch size. Default: ``64``.
        shuffle: Randomly shuffle the data. Default: ``True``.

    Returns:
        MNIST dataloader
    """
    return DataLoader(load_mnist_dataset(), batch_size=batch_size, shuffle=shuffle)


def load_one_batch_mnist(
    batch_size: int = 64, shuffle: bool = True
) -> Tuple[Tensor, Tensor]:
    """Return a single mini-batch (inputs, labels) from MNIST.

    Args:
        batch_size: Mini-batch size. Default: ``64``.
        shuffle: Randomly shuffle the data. Default: ``True``.

    Returns:
        A single batch (inputs, labels) from MNIST.
    """
    dataloader = get_mnist_dataloader(batch_size, shuffle)
    X, y = next(iter(dataloader))

    return X, y


def autograd_diag_ggn_exact(
    X: Tensor, y: Tensor, model: Module, loss_function: Module, idx: List[int] = None
) -> Tensor:
    """Compute the generalized Gauss-Newton diagonal with ``torch.autograd``.

    Args:
        X: Input to the model.
        y: Labels.
        model: The neural network.
        loss_function: Loss function module.
        idx: Indices for which the diagonal entries are computed. Default value ``None``
            computes the full diagonal.

    Returns:
        Exact GGN diagonal (flattened and concatenated).
    """
    diag_elements = [
        col[col_idx]
        for col_idx, col in _autograd_ggn_exact_columns(
            X, y, model, loss_function, idx=idx
        )
    ]

    return stack(diag_elements)


def _autograd_ggn_exact_columns(
    X: Tensor, y: Tensor, model: Module, loss_function: Module, idx: List[int] = None
) -> Iterator[Tuple[int, Tensor]]:
    """Yield exact generalized Gauss-Newton's columns computed with ``torch.autograd``.

    Args:
        X: Input to the model.
        y: Labels.
        model: The neural network.
        loss_function: Loss function module.
        idx: Indices of columns that are computed. Default value ``None`` computes all
            columns.

    Yields:
        Tuple of column index and respective GGN column (flattened and concatenated).
    """
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    D = sum(p.numel() for p in trainable_parameters)

    outputs = model(X)
    loss = loss_function(outputs, y)

    idx = idx if idx is not None else list(range(D))

    for d in idx:
        e_d = zeros(D, device=loss.device, dtype=loss.dtype)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, trainable_parameters)

        ggn_d_list = ggn_vector_product(loss, outputs, model, e_d_list)

        yield d, parameters_to_vector(ggn_d_list)

"""Test whether the graph is clear after a backward pass."""
from torch import rand, rand_like
from torch.nn import Linear, MSELoss, ReLU, Sequential

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact


def test_graph_clear() -> None:
    """Test that the graph is clear after a backward pass.

    More specifically, test that there are no saved quantities left over.
    """
    batch_size, in_dim, out_dim = 4, 5, 6
    model = extend(
        Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
    )
    inputs = rand(batch_size, in_dim)
    extension = DiagGGNExact()
    outputs = model(inputs)
    loss = extend(MSELoss())(outputs, rand_like(outputs))
    with backpack(extension):
        loss.backward()

    # test that the dictionary is empty
    saved_quantities: dict = extension.saved_quantities._saved_quantities
    assert type(saved_quantities) is dict
    assert not saved_quantities

r"""Chunking computations for large-output networks
===================================================

:math:
     \mathcal{L}(\mathbf{\theta}) = ...

Auto-encoder

First, the imports:
"""

from functools import partial
from time import time

from memory_profiler import memory_usage
from torch import allclose, manual_seed, rand, zeros_like
from torch.nn import Conv2d, ConvTranspose2d, Flatten, MSELoss, Sequential, Sigmoid

from backpack import backpack, extend
from backpack.custom_module.slicing import Slicing
from backpack.extensions import DiagGGNExact

manual_seed(0)

batch_size = 5
channels = 3
shape = (channels, 32, 32)
X = rand(batch_size, *shape)

encoder = Sequential(
    Conv2d(channels, 20, 3),
    Sigmoid(),
)
decoder = Sequential(
    ConvTranspose2d(20, channels, 3),
)
model = Sequential(
    encoder,
    decoder,
    Flatten(),
)
loss_func = MSELoss()


model = extend(model)
loss_func = extend(loss_func)

print(f"Output dimension: {model(X).shape[1]}")


def diag_ggn_exact_one_go():
    reconstruction = model(X)
    error = loss_func(reconstruction, X.flatten(start_dim=1))
    with backpack(DiagGGNExact()):
        error.backward()

    return [p.diag_ggn_exact.clone() for p in model.parameters()]


mem_usage = memory_usage(diag_ggn_exact_one_go, interval=1e-3)
print("One go")
print(f"\t Peak memory: {max(mem_usage)}")

start = time()
one_go = diag_ggn_exact_one_go()
end = time()
print(f"\tTime [s]: {end-start:.2e}")


def diag_ggn_exact_chunked(num_chunks: int):
    reconstruction = model(X)

    assert reconstruction.numel() % num_chunks == 0
    assert reconstruction.dim() == 2

    chunk_size = reconstruction.shape[1] // num_chunks

    diag_ggn_exact = [zeros_like(p) for p in model.parameters()]

    for idx in range(num_chunks):
        slicing = (slice(None), slice(idx * chunk_size, (idx + 1) * chunk_size))
        chunk_module = extend(Slicing(slicing))

        sliced_reconstruction = chunk_module(reconstruction)
        sliced_X = X.flatten(start_dim=1)[slicing]

        slice_error = loss_func(sliced_reconstruction, sliced_X)

        with backpack(DiagGGNExact(), retain_graph=True):
            slice_error.backward(retain_graph=True)

        for p_idx, p in enumerate(model.parameters()):
            diag_ggn_exact[p_idx] += p.diag_ggn_exact

    # fix normalization
    return [ggn / num_chunks for ggn in diag_ggn_exact]


num_chunks = [1, 4, 16, 64]

for n in num_chunks:
    print(f"{n} chunks:")
    mem_usage = memory_usage(partial(diag_ggn_exact_chunked, n), interval=1e-3)
    print(f"\tPeak memory: {max(mem_usage)}")

    start = time()
    chunked = diag_ggn_exact_chunked(n)
    end = time()
    print(f"\tTime [s]: {end-start:.2e}")

    correct = [
        allclose(g1, g2, rtol=5e-3, atol=5e-5) for g1, g2 in zip(one_go, chunked)
    ]
    print(f"\tCorrect: {correct}")

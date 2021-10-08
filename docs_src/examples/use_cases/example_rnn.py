"""Recurrent networks
====================
"""
# %%
# There are two different approaches to using BackPACK with RNNs.
#
# 1. :ref:`Custom RNN with BackPACK custom modules`:
#    Build your RNN with custom modules provided by BackPACK
#    without overwriting the forward pass. This approach is useful if you want to
#    understand how BackPACK handles RNNs, or if you think building a container
#    module that implicitly defines the forward pass is more elegant than coding up
#    a forward pass.
#
# 2. :ref:`RNN with BackPACK's converter`:
#    Automatically convert your model into a BackPACK-compatible architecture.
#
# .. note::
#    RNNs are still an experimental feature. Always double-check your
#    results, as done in this example! Open an issue if you encounter a bug to help
#    us improve the support.
#
#    Not all extensions support RNNs (yet). Please create a feature request in the
#    repository if the extension you need is not supported.

# %%
# Let's get the imports out of the way.
import torch
from torch import (
    allclose,
    cat,
    cuda,
    device,
    int32,
    linspace,
    manual_seed,
    nn,
    randint,
    zeros_like,
)

from backpack import backpack, extend
from backpack.custom_module.graph_utils import BackpackTracer
from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple
from backpack.extensions import BatchGrad, DiagGGNExact
from backpack.utils.examples import autograd_diag_ggn_exact

manual_seed(0)
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
DEVICE = device("cpu")  # BackPACK works perfectly fine on cuda, but autograd does not


# %%
# Define Tolstoi Char RNN from DeepOBS.
# This network is trained on Leo Tolstoi's War and Peace
# and learns to predict the next character.
class TolstoiCharRNN(nn.Module):
    def __init__(self):
        super(TolstoiCharRNN, self).__init__()
        self.batch_size = 8
        self.hidden_dim = 64
        self.num_layers = 2
        self.seq_len = 15
        self.vocab_size = 25

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_dim
        )
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.36,
            batch_first=True,
        )
        # deactivate redundant bias
        self.lstm.bias_ih_l0.data = zeros_like(self.lstm.bias_ih_l0)
        self.lstm.bias_ih_l1.data = zeros_like(self.lstm.bias_ih_l1)
        self.lstm.bias_ih_l0.requires_grad = False
        self.lstm.bias_ih_l1.requires_grad = False
        self.dense = nn.Linear(
            in_features=self.hidden_dim, out_features=self.vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, new_state = self.lstm(x)
        x = self.dropout(x)
        output = self.dense(x)
        output = output.permute(0, 2, 1)
        return output

    def input_target_fn(self):
        input = randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        # target is the input shifted by 1 in time axis
        target = cat(
            [
                randint(0, self.vocab_size, (self.batch_size, 1)),
                input[:, :-1],
            ],
            dim=1,
        )
        return input.to(DEVICE), target.to(DEVICE)

    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss().to(DEVICE)


tolstoi_char_rnn = TolstoiCharRNN().to(DEVICE).eval()
loss_function = extend(tolstoi_char_rnn.loss_fn())
x, y = tolstoi_char_rnn.input_target_fn()
# %%
# Custom RNN with BackPACK custom modules
# -------------
# Second-order extensions only work if every node in the computation graph is an
# ``nn`` module that can be extended by BackPACK. The above RNN
# :py:class:`TolstoiCharRNN<TolstoiCharRNN>` does not satisfy these conditions, because
# it has a multi-layer :py:class:`torch.nn.LSTM` and implicitly uses the getitem
# and :py:meth:`permute() <torch.Tensor.permute>`
# functions in the :py:meth:`forward() <torch.nn.Module.forward>` method.
#
# To build RNN without overwriting the forward pass, BackPACK offers custom modules:
#
# 1. :py:class:`ReduceTuple <backpack.custom_module.reduce_tuple.ReduceTuple>`
#
# 2. :py:class:`Permute <backpack.custom_module.permute.Permute>`
#
# With the above modules, we can build a simple RNN as a container that implicitly
# defines the forward pass:
tolstoi_char_rnn_equivalent = nn.Sequential(
    nn.Embedding(tolstoi_char_rnn.vocab_size, tolstoi_char_rnn.hidden_dim),
    nn.Dropout(p=0.2),
    nn.LSTM(tolstoi_char_rnn.hidden_dim, tolstoi_char_rnn.hidden_dim, batch_first=True),
    ReduceTuple(index=0),
    nn.Dropout(p=0.36),
    nn.LSTM(tolstoi_char_rnn.hidden_dim, tolstoi_char_rnn.hidden_dim, batch_first=True),
    ReduceTuple(index=0),
    nn.Dropout(p=0.2),
    nn.Linear(tolstoi_char_rnn.hidden_dim, tolstoi_char_rnn.vocab_size),
    Permute(0, 2, 1),
)

tolstoi_char_rnn_equivalent.eval().to(DEVICE)
tolstoi_char_rnn_equivalent = extend(tolstoi_char_rnn_equivalent)
loss = loss_function(tolstoi_char_rnn_equivalent(x), y)
with backpack(BatchGrad(), DiagGGNExact()):
    loss.backward()

for name, parameter in tolstoi_char_rnn_equivalent.named_parameters():
    print(
        name,
        parameter.shape,
        parameter.grad_batch.shape,
        parameter.diag_ggn_exact.shape,
    )

# %%
# Comparison with :py:mod:`torch.autograd`:
#
# .. note::
#
#    Computing the full GGN diagonal with PyTorch's built-in automatic differentiation
#    can be slow, depending on the number of parameters. To reduce run time, we only
#    compare some elements of the diagonal.
diag_ggn_exact_vector = cat(
    [p.diag_ggn_exact.flatten() for p in tolstoi_char_rnn_equivalent.parameters()]
)

num_params = sum(p.numel() for p in tolstoi_char_rnn_equivalent.parameters())
num_to_compare = 10
idx_to_compare = linspace(0, num_params - 1, num_to_compare, device=DEVICE, dtype=int32)

with torch.backends.cudnn.flags(enabled=False):
    diag_ggn_exact_to_compare = autograd_diag_ggn_exact(
        x, y, tolstoi_char_rnn_equivalent, loss_function, idx=idx_to_compare
    )

print("Do the exact GGN diagonals match?")
for idx, element in zip(idx_to_compare, diag_ggn_exact_to_compare):
    match = allclose(element, diag_ggn_exact_vector[idx], atol=1e-10)
    print(f"Diagonal entry {idx:>8}: {match}:\t{element}, {diag_ggn_exact_vector[idx]}")
    if not match:
        raise AssertionError("Exact GGN diagonals don't match!")

# %%
# RNN with BackPACK's converter
# -------------
# If you are not building a RNN through custom modules but for instance want to
# directly use the Tolstoi Char RNN, BackPACK offers a converter.
# It analyzes the model and tries to turn it into a compatible architecture. The result
# is a :py:class:`torch.fx.GraphModule` that exclusively consists of modules.
#
# Here, we demo the converter on above Tolstoi Char RNN.
#
# Let's convert it in the call to :py:func:`extend <backpack.extend>`:

# use BackPACK's converter to extend the model (turned off by default)
tolstoi_char_rnn = extend(tolstoi_char_rnn, use_converter=True)

# %%
# To get an understanding what happened, we can inspect the model's graph with the
# following helper function:


def print_table(module: nn.Module) -> None:
    """Prints a table of the module.

    Args:
        module: module to analyze
    """
    graph = BackpackTracer().trace(module)
    graph.print_tabular()


print_table(tolstoi_char_rnn)

# %%
# Note that the computation graph fully consists of modules (indicated by
# ``call_module`` in the first table column) such that BackPACK's hooks can
# successfully backpropagate additional information for its second-order extensions
# (first-order extensions work, too).
#
# Let's verify that second-order extensions are working:
output = tolstoi_char_rnn(x)
loss = loss_function(output, y)

with backpack(DiagGGNExact()):
    loss.backward()

for name, parameter in tolstoi_char_rnn.named_parameters():
    if parameter.requires_grad:
        print(f"{name}'s diag_ggn_exact: {parameter.diag_ggn_exact.shape}")

diag_ggn_exact_vector = cat(
    [
        p.diag_ggn_exact.flatten()
        for p in tolstoi_char_rnn.parameters()
        if p.requires_grad
    ]
)

# %%
# Comparison with :py:mod:`torch.autograd`:
#
# .. note::
#
#    Computing the full GGN diagonal with PyTorch's built-in automatic differentiation
#    can be slow, depending on the number of parameters. To reduce run time, we only
#    compare some elements of the diagonal.

num_params = sum(p.numel() for p in tolstoi_char_rnn.parameters() if p.requires_grad)
num_to_compare = 10
idx_to_compare = linspace(0, num_params - 1, num_to_compare, device=DEVICE, dtype=int32)

with torch.backends.cudnn.flags(enabled=False):
    diag_ggn_exact_to_compare = autograd_diag_ggn_exact(
        x, y, tolstoi_char_rnn, loss_function, idx=idx_to_compare
    )

print("Do the exact GGN diagonals match?")
for idx, element in zip(idx_to_compare, diag_ggn_exact_to_compare):
    match = allclose(element, diag_ggn_exact_vector[idx], atol=1e-10)
    print(f"Diagonal entry {idx:>8}: {match}:\t{element}, {diag_ggn_exact_vector[idx]}")
    if not match:
        raise AssertionError("Exact GGN diagonals don't match!")

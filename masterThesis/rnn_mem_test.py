"""This script test RNNs.

Whether the results of all layers are saved
during forward pass of an multilayer RNN.
To this the memory usage of RNN(num_layers) is compared with num_layers*RNN.
"""

import time

import matplotlib.pyplot as plt
import torch
from memory_profiler import memory_usage


num_layers = 1
input_size = 2
hidden_size = 100
seq_len = 10
batch_size = 400000


def _sleep():
    time.sleep(0.02)


def rnn_stacked():
    """Create a stacked rnn."""
    input_rand = torch.randn(seq_len, batch_size, input_size)
    input_rand.requires_grad = True
    RNN_stacked = torch.nn.RNN(input_size, hidden_size, num_layers)
    _sleep()
    output, _ = RNN_stacked.forward(input_rand)
    output.sum().backward()
    _sleep()
    del RNN_stacked, output, _, input_rand


def rnn_list():
    """Create a list of rnns."""
    input_rand = torch.randn(seq_len, batch_size, input_size)
    input_rand.requires_grad = True
    RNN_list = torch.nn.ModuleList()
    for i in range(num_layers):
        input_size_temp = input_size if i == 0 else hidden_size
        RNN_list.append(torch.nn.RNN(input_size_temp, hidden_size, 1))
    _sleep()
    output = input_rand
    for layer in RNN_list:
        output, _ = layer.forward(output)
    output.sum().backward()
    _sleep()
    del RNN_list, output, _, input_rand


def _report_memory(f, interval=0.001):
    """Print memory statistics of a function execution.

    Args:
        f (function): Function to evaluate.
        interval (float): Defaults to 0.001.

    Returns:
        torch.Tensor
    """
    start = time.time()
    mem_usage = torch.tensor(memory_usage(f, interval=interval))
    end = time.time()
    print("Memory consumption:")
    print("-------------------")
    print(f"Min   : {mem_usage.min():.1f} MB")
    print(f"Max   : {mem_usage.max():.1f} MB")
    print(f"Mean  : {mem_usage.mean():.1f} MB")
    print(f"Median: {torch.median(mem_usage):.1f} MB")
    print(f"\nTime: {end - start:.2f} s")
    return mem_usage


def _plot_memory(mem_usage_list, labels):
    """Create plot from list of memory consumption.

    Args:
        mem_usage_list (list): memory usage data
        labels (list): labels of data
    """
    fig, ax = plt.subplots()
    for i, mem_usage in enumerate(mem_usage_list):
        ax.plot(range(len(mem_usage)), mem_usage, label=labels[i])
    fig.legend()
    plt.show()


if __name__ == "__main__":
    mem_usage_list = []
    n_repeat = 2
    labels = [f"RNN stacked {i}" for i in range(n_repeat)] + [
        f"RNN list {i}" for i in range(n_repeat)
    ]
    for _ in range(n_repeat):
        mem_usage_list.append(_report_memory(rnn_stacked))
        mem_usage_list.append(_report_memory(rnn_list))
    _plot_memory(mem_usage_list, labels=labels)

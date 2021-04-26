import time

import matplotlib.pyplot as plt
import torch
from memory_profiler import memory_usage

"""
This script test whether the results of all layers are saved 
during forward pass of an multilayer RNN.
To this the memory usage of RNN(num_layers) is compared with num_layers*RNN.
"""

num_layers = 1
input_size = 2
hidden_size = 100
seq_len = 10
batch_size = 400000

input_rand = torch.randn(seq_len, batch_size, input_size)
input_rand.requires_grad = True


def sleep():
    time.sleep(0.02)


def rnn_stacked():
    """Create a stacked rnn."""
    RNN_stacked = torch.nn.RNN(input_size, hidden_size, num_layers)
    sleep()
    output, _ = RNN_stacked.forward(input_rand)
    sleep()
    del RNN_stacked, output, _


def rnn_list():
    RNN_list = torch.nn.ModuleList()
    for i in range(num_layers):
        input_size_temp = input_size if i == 0 else hidden_size
        RNN_list.append(torch.nn.RNN(input_size_temp, hidden_size, 1))
    sleep()
    output = input_rand
    for layer in RNN_list:
        output, _ = layer.forward(output)
    sleep()
    del RNN_list, output, _


def report_memory(f, interval=0.001):
    """Print memory statistics of a function execution."""
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


def plot_memory(mem_usage_list, labels):
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
        mem_usage_list.append(report_memory(rnn_stacked))
        mem_usage_list.append(report_memory(rnn_list))
    plot_memory(mem_usage_list, labels=labels)

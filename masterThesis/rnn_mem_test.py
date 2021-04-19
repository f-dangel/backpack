import torch
import sys

from pytorch_memlab import LineProfiler
"""
This script test whether the results of all layers are saved 
during forward pass of an multilayer RNN.
To this the memory usage of RNN(num_layers) is compared with num_layers*RNN.
"""


num_layers = 10
input_size = 5
hidden_size = 5
RNN_stacked = torch.nn.RNN(input_size, hidden_size, num_layers)
RNN_list = torch.nn.ModuleList()
for i in range(num_layers):
    input_size = input_size if i == 0 else hidden_size
    RNN_list.append(torch.nn.RNN(input_size, hidden_size, 1))

seq_len = 10
batch_size = 1000
input_rand = torch.randn(seq_len, batch_size, input_size)
input_rand.requires_grad = True

RNN_stacked.forward(input_rand)


print(sys.getsizeof(RNN_stacked))

import torch

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


# data loading
import torchvision
import torchvision.transforms as transforms
from os import path

# layers
import torch
from torch.nn import CrossEntropyLoss
from bpexts.hbp.linear import HBPLinear

# for HBP
from bpexts.hbp.loss import batch_summed_hessian

# optimizer
from bpexts.optim.cg_newton import CGNewton

# auxiliary
from bpexts.utils import set_seeds

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set_seeds(0)

batch_size = 500

# download directory
data_dir = '~/tmp/MNIST'

# training set loader
train_set = torchvision.datasets.MNIST(
    root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

# layer parameters
in_features = 784
out_features = 10
bias = True

# linear layer
model = HBPLinear(in_features=in_features, out_features=out_features, bias=bias)
# load to device
model.to(device)
print(model)

loss_func = CrossEntropyLoss()

# learning rate
lr = 0.1

# regularization
alpha = 0.02

# convergence criteria for CG
cg_maxiter = 50
cg_atol = 0.
cg_tol = 0.1

# construct the optimizer
optimizer = CGNewton(
    model.parameters(),
    lr=lr,
    alpha=alpha,
    cg_atol=cg_atol,
    cg_tol=cg_tol,
    cg_maxiter=cg_maxiter)

# use the GGN
modify_2nd_order_terms = 'zero'

# train for two epochs
num_epochs = 2

# log some metrics
train_epoch = [ ] 
batch_loss = [ ]
batch_acc = [ ]

samples = 0
samples_per_epoch = 60000.
for epoch in range(num_epochs):
    iters = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        # reshape and load to device
        images = images.reshape(-1, in_features).to(device)
        labels = labels.to(device)

        # 1) forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # set gradients to zero
        optimizer.zero_grad()

        # Hessian backpropagation and backward pass
        # 2) batch average of Hessian of loss w.r.t. model output
        output_hessian = batch_summed_hessian(loss, outputs)
        # 3) compute gradients
        loss.backward()
        # 4) propagate Hessian back through the graph
        model.backward_hessian(
            output_hessian, modify_2nd_order_terms=modify_2nd_order_terms)

        # 5) second-order optimization step
        optimizer.step()

        # compute statistics
        total = labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # update lists
        samples += total
        train_epoch.append(samples / samples_per_epoch)
        batch_loss.append(loss.item())
        batch_acc.append(accuracy)

        # print every 5 iterations
        if i % 5 == 0:
            print(
                'Epoch [{}/{}], Iter. [{}/{}], Loss: {:.4f}, Acc.: {:.4f}'.
                format(epoch + 1, num_epochs, i + 1, iters, loss.item(),
                       accuracy))

# plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.subplots(121, figsize=(7,3))

# plot batch loss
plt.subplot(121)
plt.plot(train_epoch, batch_loss, color='darkorange')
plt.xlabel('epoch')
plt.ylabel('batch loss')

# plot batch accuracy
plt.subplot(122)
plt.plot(train_epoch, batch_acc, color='darkblue')
plt.xlabel('epoch')
plt.ylabel('batch accuracy')

# save plot
plt.tight_layout()
plt.savefig('01_single_layer_mnist_metrics.png')

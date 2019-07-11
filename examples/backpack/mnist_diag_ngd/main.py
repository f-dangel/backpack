from torch import no_grad
import torch
from torch.autograd.profiler import profile

from data_loader import get_data_loaders
from model import make_model, make_lossfunc, make_quadratic_model
from optim import get_sgd, get_ngd

PROFILE = False

TEST_BATCH_SIZE = 512
BATCH_SIZE = 128
SEED = 0
EPOCHS = 2
LR = .1

USE_SGD = True

torch.manual_seed(SEED)

model = make_model()
#model = make_quadratic_model()

lossfunc = make_lossfunc()
train_loader, test_loader = get_data_loaders(BATCH_SIZE, TEST_BATCH_SIZE)

if USE_SGD:
    optimizer = get_sgd(model, LR)
else:
    optimizer = get_ngd(model)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if USE_SGD:
            def loss_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = lossfunc(output, target)
                loss.backward()
                return loss
        else:
            def loss_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = lossfunc(output, target)
                return loss, output

        if PROFILE:
            with profile() as prof:
                loss = optimizer.step(loss_closure)

            print(prof.table("cpu_time"))
            import pdb

            pdb.set_trace()
        else:
            loss = optimizer.step(loss_closure)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            batch_idx,
            len(train_loader),
            100. * batch_idx / len(train_loader),
            loss.item()
        ))
        if batch_idx > 100:
            break

    model.eval()
    correct = 0
    with no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("-" * 20)
    print('Test set accuracy: {}/{} ({:.0f}%)'.format(
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    print("-" * 20)

from torch import no_grad
import torch
from torch.autograd.profiler import profile

from data_loader import get_data_loaders
from model import make_model, make_lossfunc
from optim import get_optimizer

PROFILE = False

TEST_BATCH_SIZE = 512
BATCH_SIZE = 16
SEED = 0
EPOCHS = 2
LR = 0.01

torch.manual_seed(SEED)

model = make_model()
lossfunc = make_lossfunc()
train_loader, test_loader = get_data_loaders(BATCH_SIZE, TEST_BATCH_SIZE)
optimizer = get_optimizer(model, LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()


        def loss_closure():
            loss = lossfunc(model(data), target)
            loss.backward()
            return loss


        if PROFILE:
            with profile() as prof:
                optimizer.step(loss_closure)

            print(prof.table("cpu_time"))
            import pdb

            pdb.set_trace()
        else:
            optimizer.step(loss_closure)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            batch_idx,
            len(train_loader),
            100. * batch_idx / len(train_loader),
            loss.item()
        ))
        if batch_idx > 3:
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

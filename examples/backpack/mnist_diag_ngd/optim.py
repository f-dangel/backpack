from torch.optim.sgd import SGD

from bpexts.optim.diagggn_curvature_wrapper import DiagGGNCurvatureWrapper
from bpexts.optim.fancy_damping import FancyDampingWrapper


def get_ngd(model):
    return FancyDampingWrapper(
        model.parameters(),
        DiagGGNCurvatureWrapper(list(model.parameters())),
#        inv_damping=150.
    )

def get_sgd(model, lr=0.1):
    return SGD(model.parameters(), lr=lr)




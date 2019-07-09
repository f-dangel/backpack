
from bpexts.optim.diagggn_curvature_wrapper import DiagGGNCurvatureWrapper
from bpexts.optim.fancy_damping import FancyDampingWrapper


def get_optimizer(model, lr):
    return FancyDampingWrapper(
        model.parameters(),
        DiagGGNCurvatureWrapper(list(model.parameters())),
        inv_damping=1500.
    )




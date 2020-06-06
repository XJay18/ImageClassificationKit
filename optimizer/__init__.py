from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop

OPTIMIZERS = {
    "SGD": SGD,
    "Adam": Adam,
    "RMSprop": RMSprop
}


def fetch_optimizer(name="SGD"):
    print("Using optimizer: %s." % name)
    return OPTIMIZERS[name]

from .cifar import WrappedCIFAR10
from .cifar import WrappedCIFAR100
from .quickdraw import WrappedQuickDraw

LOADERS = {
    "WrappedCIFAR10": WrappedCIFAR10,
    "WrappedCIFAR100": WrappedCIFAR100,
    "WrappedQuickDraw": WrappedQuickDraw,
}


def load_dataset(name):
    print("Loading dataset: %s......" % name)
    return LOADERS[name]

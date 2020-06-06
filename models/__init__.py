from .resnet import ResNetMini, ResNet18
from .resnext import ResNextMini
from .densenet import DenseNetMini, DenseNet121
from .common import IntermediateLayerGetter

MODELS = {
    "ResNetMini": ResNetMini,
    "ResNet18": ResNet18,
    "ResNextMini": ResNextMini,
    "DenseNetMini": DenseNetMini,
    "DenseNet121": DenseNet121
}


def load_model(name):
    print("Using model: %s." % name)
    return MODELS[name]

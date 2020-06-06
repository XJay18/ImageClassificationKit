from torchvision.models.densenet import DenseNet
from models.common import apply_init


class DenseNetMini(DenseNet):
    def __init__(self, num_classes=10, init=None):
        super(DenseNetMini, self).__init__(
            growth_rate=12, block_config=(3, 6, 12, 8),
            num_init_features=64, num_classes=num_classes
        )
        apply_init(init, self.modules())


class DenseNet121(DenseNet):
    def __init__(self, num_classes=10, init=None):
        super(DenseNet121, self).__init__(
            growth_rate=32, block_config=(6, 12, 24, 16),
            num_init_features=64, num_classes=num_classes
        )
        apply_init(init, self.modules())

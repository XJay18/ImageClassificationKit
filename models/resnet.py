import torch
import torch.nn as nn
from models.common import apply_init
from torchvision.models.resnet import resnet18


class ResBlock(nn.Module):
    def __init__(self, in_channels, depth, groups=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, depth, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=depth)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            depth, depth, kernel_size=3, groups=groups,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=depth)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, depth, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(num_features=depth)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetMini(nn.Module):
    """
    ResNet used for evaluating CIFAR-10 dataset described in
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, depth_list=(16, 32, 64), num_classes=10, init=None):
        super(ResNetMini, self).__init__()

        self.conv1 = nn.Conv2d(1, depth_list[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_features=depth_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.block1 = self._build_block(ResBlock, depth_list[0], depth_list[0])
        self.block2 = self._build_block(ResBlock, depth_list[0], depth_list[1], 2)
        self.block3 = self._build_block(ResBlock, depth_list[1], depth_list[2], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(depth_list[2], num_classes)
        # apply the weights initiator
        apply_init(init, self.modules())

    @staticmethod
    def _build_block(block, in_channels, depth, stride=1):
        return nn.Sequential(
            block(in_channels, depth, stride=stride),
            block(depth, depth),
            block(depth, depth)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, init=None):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained, num_classes=num_classes)
        # apply the weights initiator
        apply_init(init, self.model.modules())

    def __str__(self):
        return str(self.model)

    def forward(self, x):
        return self.model.forward(x)


if __name__ == '__main__':
    def run_model_gpu(model):
        print("Running model on GPU.....")
        resnet = model.cuda()
        # print(resnet)
        tensor = torch.randn([2, 3, 32, 32]).cuda()
        y_pre = resnet(tensor)
        print(y_pre.shape)


    def run_model_cpu(model):
        print("Running model on CPU.....")
        resnet = model
        print(resnet)
        tensor = torch.randn([2, 3, 32, 32])
        y_pre = resnet(tensor)
        print(y_pre.shape)


    m = ResNetMini()
    # m = ResNet18()
    run_model_gpu(m)
    # run_model_cpu()

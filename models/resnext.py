import torch
import torch.nn as nn
from models.common import apply_init
from models.resnet import ResNetMini


class ResNextBlock(nn.Module):
    def __init__(self, in_channels, depth, groups=1, stride=1):
        super(ResNextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, depth, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(depth)
        self.conv2 = nn.Conv2d(
            depth, depth * groups, 3, stride=1,
            padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(depth * groups)
        self.conv3 = nn.Conv2d(depth * groups, depth, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNextMini(ResNetMini):
    def __init__(self, depth_list=(16, 32, 64), num_classes=10, init=None):
        super(ResNextMini, self).__init__(depth_list, num_classes, init)

        self.block1 = self._build_block(ResNextBlock, depth_list[0], depth_list[0])
        self.block2 = self._build_block(ResNextBlock, depth_list[0], depth_list[1], 2)
        self.block3 = self._build_block(ResNextBlock, depth_list[1], depth_list[2], 2)

        apply_init(init, self.modules())

    @staticmethod
    def _build_block(block, in_channels, depth, groups=4, stride=1):
        return nn.Sequential(
            block(in_channels, depth, groups=groups, stride=stride),
            block(depth, depth, groups=groups),
            block(depth, depth, groups=groups)
        )


if __name__ == '__main__':
    def run_model_gpu():
        print("Running model on GPU.....")
        resnext = ResNextMini().cuda()
        print(resnext)
        tensor = torch.randn([2, 3, 32, 32]).cuda()
        y_pre = resnext(tensor)
        print(y_pre.shape)


    def run_model_cpu():
        print("Running model on CPU.....")
        resnext = ResNextMini()
        print(resnext)
        tensor = torch.randn([2, 3, 32, 32])
        y_pre = resnext(tensor)
        print(y_pre.shape)


    run_model_gpu()
    # run_model_cpu()
    # print(nn.Conv2d(32, 32, kernel_size=3, stride=1, groups=4))

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Pad, RandomCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor, Normalize

CIFAR10_CLASS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class WrappedCIFAR10(CIFAR10):
    def __init__(self, root, train=True, download=False, *args, **kwargs):
        self.categories = CIFAR10_CLASS
        if train:
            transforms = Compose([
                Pad(padding=4),
                RandomCrop(size=32),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        super(WrappedCIFAR10, self).__init__(
            root=root, train=train, transform=transforms,
            download=download, *args, **kwargs)


class WrappedCIFAR100(CIFAR100):
    def __init__(self, root, train=True, download=False, *args, **kwargs):
        if train:
            transforms = Compose([
                Pad(padding=4),
                RandomCrop(size=32),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        super(WrappedCIFAR100, self).__init__(
            root=root, train=train, transform=transforms,
            download=download, *args, **kwargs)


if __name__ == '__main__':
    # You need to change the argument according to your own settings.
    ROOT = "path/to/data"


    def download_CIFAR10():
        print("Downloading CIFAR10......")
        CIFAR = WrappedCIFAR10(root=ROOT, download=True)


    def test_CIFAR10():
        CIFAR = WrappedCIFAR10(root=ROOT, train=True)
        CIFAR_dataloader = DataLoader(CIFAR, batch_size=4, shuffle=True, num_workers=4)

        for i in CIFAR_dataloader:
            print(i[0].shape, i[1])


    def test_CIFAR10_using_network():
        from torch.utils import data
        from models.resnet import ResNetMini
        from loss import fetch_loss
        from optimizer import fetch_optimizer

        batch_size = 4
        epoch = 1

        train_data = WrappedCIFAR10(root=ROOT, train=True)
        train_loader = data.DataLoader(
            train_data, batch_size=batch_size,
            shuffle=True, num_workers=4)

        network = ResNetMini().cuda()
        loss_func = fetch_loss()

        optim = fetch_optimizer()(network.parameters(), lr=0.001, momentum=0.9)

        for e in range(epoch):
            for iter, data in enumerate(train_loader):
                I, Y = data
                I, Y = I.cuda(), Y.cuda()
                Y_pre = network(I)
                loss = loss_func(Y_pre, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if iter % 100 == 0:
                    print("iter: ", iter, " , loss: ", loss.item())


    # download_CIFAR10()
    # test_CIFAR10()
    # test_CIFAR10_using_network()

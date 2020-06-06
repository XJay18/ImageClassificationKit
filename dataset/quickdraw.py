import os
from urllib import request

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Pad, RandomCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor
from tqdm import tqdm

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

np.random.seed(0)


class QuickDraw(Dataset):
    def __init__(self, root, train=True, download=False, process=False, transforms=None):
        super(QuickDraw, self).__init__()
        if download:
            self.download(root)
        if process:
            self.process(root)
        self.x, self.y = self.fetch_split(root, train)
        self.transforms = transforms

    @staticmethod
    def download(root, start=0):
        """
        Download the QuickDraw data set.

        Args:
            root: The path of QuickDraw on your machine, e.g., /path/to/QuickDraw.
            start: The index of the class in the data set to begin with.
        """
        path = os.path.join(root, "QuickDraw", "raw")
        if not os.path.exists(path):
            os.makedirs(path)
        cls = CLS[start:]
        cls_gen = enumerate(tqdm(cls, ncols=100, position=0, leave=True))
        # download
        print("Downloading QuickDraw Dataset...\n")
        for idx, c in cls_gen:
            url = BASE + c.replace("_", "%20") + ".npy"
            cpath = os.path.join(path, c + ".npy")
            request.urlretrieve(url, cpath)
        print("\nDone.")

    @staticmethod
    def process(root, train_per_cls=10000, val_per_cls=1000):
        """
        Process the downloaded files to npz format.

        Args:
            root: The path of QuickDraw on your machine, e.g., /path/to/QuickDraw.
            train_per_cls: The number of training samples per class.
            val_per_cls: The number of validation samples per class.
        """
        qdata = os.path.join(root, "QuickDraw", "raw")
        target = os.path.join(root, "QuickDraw", "processed")
        if not os.path.exists(target):
            os.makedirs(target)

        print("Processing training and validation data...\n")
        train_x = list()
        train_y = list()
        val_x = list()
        val_y = list()
        files = os.listdir(qdata)
        tqdm_files = tqdm(files, ncols=100, position=0, leave=True)
        files_gen = enumerate(tqdm_files)
        for idx, f in files_gen:
            data = np.load(os.path.join(qdata, f))
            name, _ = os.path.splitext(f)

            # choosing from 0~99999 to construct train set
            indices = np.arange(0, 100000)
            indices = np.random.choice(
                indices, train_per_cls, replace=False
            )
            train_data = data[indices]
            train_label = np.full(train_data.shape[0], MAPPING[name])
            # append the data
            train_x.append(train_data.reshape([-1, 28, 28]))
            train_y.append(train_label)

            # choosing from 100000~109999 to construct val set
            indices = np.arange(100000, 110000)
            indices = np.random.choice(
                indices, val_per_cls, replace=False
            )
            val_data = data[indices]
            val_label = np.full(val_data.shape[0], MAPPING[name])
            # append the data
            val_x.append(val_data.reshape([-1, 28, 28]))
            val_y.append(val_label)

        train_x = np.vstack(train_x)
        train_y = np.hstack(train_y)
        val_x = np.vstack(val_x)
        val_y = np.hstack(val_y)

        # randomize the dataset
        permutation = np.random.permutation(train_y.shape[0])
        train_x = train_x[permutation, :, :]
        train_y = train_y[permutation]
        permutation = np.random.permutation(val_y.shape[0])
        val_x = val_x[permutation, :, :]
        val_y = val_y[permutation]
        print("\nDone.")

        print("\nSaving processed data as numpy binary files...")
        np.savez_compressed(target + "/train", data=train_x, target=train_y)
        np.savez_compressed(target + "/val", data=val_x, target=val_y)
        print("\nDone. Files are now available in directory: %s" % target)

    @staticmethod
    def fetch_split(root, train=True):
        """
        Load the npz format data.

        Args:
            root: The path of QuickDraw on your machine, e.g., /path/to/QuickDraw.
            train: Whether to load train split or validation split.
        """
        target = os.path.join(root, "QuickDraw", "processed")
        train_path = os.path.join(target, "train.npz")
        val_path = os.path.join(target, "val.npz")
        if train and os.path.exists(train_path):
            data = np.load(train_path)["data"]
            label = np.load(train_path)["target"]
        elif not train and os.path.exists(val_path):
            data = np.load(val_path)["data"]
            label = np.load(val_path)["target"]
        else:
            raise FileNotFoundError("train.npz or val.npz file not found.")
        label = torch.tensor(label, dtype=torch.int64)
        return data, label

    def __getitem__(self, item):
        img = self.x[item]
        img = Image.fromarray(img, "L")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.y[item]

    def __len__(self):
        return self.y.shape[0]


class WrappedQuickDraw(QuickDraw):
    def __init__(self, root, train=True, download=False, process=False):
        transforms = Compose([
            Pad(padding=4),
            RandomCrop(size=32),
            RandomHorizontalFlip(p=0.5),
            ToTensor()
        ])
        super(WrappedQuickDraw, self).__init__(
            root, train, download, process, transforms=transforms)


if __name__ == '__main__':
    # You need to change the argument according to your own settings.
    ROOT = "path/to/data"

    with open("./quickdraw.txt", "r") as f:
        CLS = f.readlines()
    CLS = [c.replace("\n", "").replace(" ", "_") for c in CLS]
    MAPPING = {j: i for i, j in enumerate(CLS)}

    # First, please use this function to download the QuickDraw data set
    # to your machine.
    def test_dataset_download():
        QuickDraw.download(ROOT, start=0)

    # Then, run the following function to process the downloaded data.
    def test_dataset_process():
        QuickDraw.process(ROOT)

    # Finally, run the following function to check whether everything is done.
    def test_wrap_loader():
        qdata = WrappedQuickDraw(ROOT, train=True)
        print("len: ", len(qdata))
        loader = DataLoader(qdata, batch_size=4, shuffle=True, num_workers=4)
        for _ in loader:
            x, y = _
            print(x.shape, " \t", y)


    # test_dataset_download()
    # test_dataset_process()
    # test_wrap_loader()

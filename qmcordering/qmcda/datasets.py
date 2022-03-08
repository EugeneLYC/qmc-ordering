import torch
import math
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np

# from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import torchvision.datasets as datasets
import qmcpy as qp

def unit_interval_to_categorical(x, K):
    c = int(math.floor(K * float(x)))
    if c >= K:
        return K-1
    elif c < 0:
        return 0
    else:
        return c

class CIFAR10:
    def __init__(self, cifar10: datasets.CIFAR10, transform, args) -> None:
        self.cifar10 = cifar10
        self.transform = transform
        self.state = 0
        self.args = args
        self.sobolengs = torch.quasirandom.SobolEngine(dimension=3, scramble=True).draw(65536)
        self.sample_visited = {i:0 for i in range(50000)}
    
    def update(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index: int):
        x = self.sobolengs[index + self.epoch]
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)

        (img,target) = self.cifar10.__getitem__(index)
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar10.data)

class CIFAR100:
    def __init__(self, cifar100: datasets.CIFAR100, transform, args) -> None:
        self.cifar100 = cifar100
        self.transform = transform
        self.state = 0
        self.args = args
        self.sobolengs = torch.quasirandom.SobolEngine(dimension=4, scramble=True).draw(65536)
        self.sample_visited = {i:0 for i in range(50000)}
    
    def update(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index: int):
        x = self.sobolengs[index + self.epoch]
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)
        angle = -15 + x[3] * 30

        (img,target) = self.cifar100.__getitem__(index)
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        img = F.rotate(img, angle)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar100.data)
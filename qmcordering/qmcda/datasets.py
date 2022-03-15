import os
import six
import math
import lmdb
import pickle

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import functional as F

from ..constants import _MAX_SOBOL_SEQ_LEN_

def unit_interval_to_categorical(x, K):
    c = int(math.floor(K * float(x)))
    if c >= K:
        return K-1
    elif c < 0:
        return 0
    else:
        return c

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class QMCDataset:
    def __init__(self,
                dataset,
                transforms,
                qmc_quotas,
                args) -> None:
        self.dataset = dataset
        assert self.dataset.transform is None
        self.transforms = transforms
        self.args = args
        self.size = self.__len__()
        self.batch_per_epoch = math.ceil(self.size / self.args.batch_size)
        
        self.qmc_dimension = sum(qmc_quotas)

        self.seq_len = 2**(int(math.ceil(
            math.log(self.size + self.args.epochs, 2)
        )))
        assert _MAX_SOBOL_SEQ_LEN_ > self.args.epochs
        self.seq_len = min(_MAX_SOBOL_SEQ_LEN_, self.seq_len)
        self.need_hash = self.size + self.args.epochs > self.seq_len
        if self.need_hash:
            self.hash_base = self.seq_len - self.args.epochs
        self.sobolseq = torch.quasirandom.SobolEngine(
            dimension=self.qmc_dimension, scramble=args.scramble).draw(self.seq_len)

        self.cur_batch = 0
        self.epoch = args.start_epoch
    
    def update_sobol(self):
        self.cur_batch += 1
        if self.cur_batch == self.batch_per_epoch:
            self.cur_batch = 0
            self.epoch += 1
    
    def __getitem__(self, index: int):
        qmc_index = index % self.hash_base if self.need_hash else index
        x = self.sobolseq[qmc_index + self.epoch].tolist()

        (img, target) = self.dataset.__getitem__(index)
        if self.transforms is not None:
            img = self.transforms(img, x)
        return (img, target)
    
    def __len__(self) -> int:
        return len(self.dataset.data)

class CIFAR10:
    def __init__(self, cifar10: datasets.CIFAR10, transform, args) -> None:
        self.cifar10 = cifar10
        self.transform = transform
        self.state = 0
        self.args = args
        self.sobolengs = torch.quasirandom.SobolEngine(dimension=3, scramble=True).draw(65536)
    
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


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class ImageNet:
    def __init__(self, imagenet: ImageFolderLMDB, transform, args) -> None:
        self.imagenet = imagenet
        self.transform = transform
        self.state = 0
        self.args = args
        self.sobolengs = torch.quasirandom.SobolEngine(dimension=5, scramble=True).draw(65536)
    
    def update(self, epoch):
        self.epoch = epoch
    
    def __getitem__(self, index: int):
        x = self.sobolengs[index + self.epoch]
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)
        angle = -15 + x[3] * 30

        (img,target) = self.imagenet.__getitem__(index)
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
        return len(self.imagenet.data)
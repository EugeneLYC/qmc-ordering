import os
import six
import math
import lmdb
import pickle

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import functional as F

from constants import _MAX_SOBOL_SEQ_LEN_

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

class Dataset:
    def __init__(self,
                dataset,
                train,
                args) -> None:
        self.dataset = dataset
        self.train = train
        assert self.dataset.transform is None
        self.args = args
        self.size = self.__len__()
        self.batch_per_epoch = math.ceil(self.size / self.args.batch_size)

        self._setup_transforms()
        self._setup_sobol_engine()

        self.cur_batch = 0
        self.epoch = self.args.start_epoch
    
    def _sanity_check(self, config):
        pass
    
    def _setup_transforms(self):
        config = json.load(
            open(self.args.transforms_json, 'r', encoding='utf-8'))
        self._sanity_check(config)
        self.use_qmc = True
        if 'use_qmc' not in config.keys() or 'use_qmc' in config.keys() and config['use_qmc'] is False:
            self.use_qmc = False
        if self.train and self.use_qmc:
            from utils import get_qmc_transforms
            self.transforms, self.qmc_quotas = get_qmc_transforms(config['train_transforms'])
            self.qmc_dimension = sum(self.qmc_quotas)
        else:
            from utils import get_uniform_transforms
            self.transforms = get_uniform_transforms(config['test_transforms'])
    
    def _setup_sobol_engine(self):
        self.seq_len = 2**(int(math.ceil(
            math.log(self.size + self.args.epochs, 2)
        )))
        assert _MAX_SOBOL_SEQ_LEN_ > self.args.epochs
        self.seq_len = min(_MAX_SOBOL_SEQ_LEN_, self.seq_len)
        self.need_hash = self.size + self.args.epochs > self.seq_len
        if self.need_hash:
            self.hash_base = self.seq_len - self.args.epochs
        self.sobolseq = torch.quasirandom.SobolEngine(dimension=self.qmc_dimension,
                                                    scramble=args.scramble).draw(self.seq_len)

    def update_sobol(self):
        self.cur_batch += 1
        if self.cur_batch == self.batch_per_epoch:
            self.cur_batch = 0
            self.epoch += 1

    def __getitem__(self, index: int):
        (img, target) = self.dataset.__getitem__(index)
        if self.use_qmc:
            qmc_index = index % self.hash_base if self.need_hash else index
            x = self.sobolseq[qmc_index + self.epoch].tolist()
            if self.transforms is not None:
                img = self.transforms(img, x)
            self.update_sobol()
        else:
            if self.transforms is not None:
                img = self.transforms(img)
        return (img, target)

    def __len__(self) -> int:
        return self.dataset.__len__()

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

        # im2arr = np.array(img)
        # im2arr = torch.from_numpy(im2arr)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

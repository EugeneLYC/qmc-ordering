# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .constants import _RANDOM_RESHUFFLING_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_, \
        _CIFAR10_, _CIFAR100_, _MNIST_, _IMAGENET_

from .qmcda.utils import get_transforms
from .qmcda.datasets import QMCDataset, ImageFolderLMDB


# The baseline data augmentation is taken from the following open source repos:
# cifar10: https://github.com/akamaster/pytorch_resnet_cifar10
# cifar100: https://github.com/weiaicunzai/pytorch-cifar100
# Currently, only CIFAR10, CIFAR100, and MNIST are supported

def _get_cifar10_loaders(args, data_path, shuffle_flag):
    loaders = {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # use a sobol sequence that does sample index aware augmentation
    if args.use_qmc_da:
        qmc_transforms, qmc_quotas = get_transforms(args)
        cifar10 = QMCDataset(dataset=datasets.CIFAR10(root=data_path, train=True, download=True),
            transforms=qmc_transforms, qmc_quotas=qmc_quotas, args=args)
        # loaders['trainset'] = cifar10
        loaders['train'] = torch.utils.data.DataLoader(
            cifar10,
            batch_size=args.batch_size, shuffle=shuffle_flag,
            persistent_workers=False,
            num_workers=args.workers, pin_memory=False)
    else:
        # use the original data augmentation
        if args.use_uniform_da:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])
        # no data augmentation in the training set
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        loaders['train'] = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True),
            batch_size=args.batch_size, shuffle=shuffle_flag,
            num_workers=args.workers, pin_memory=False)

    loaders['val'] = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return loaders

def _get_cifar100_loaders(args, data_path, shuffle_flag):
    loaders = {}
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.268, 0.257, 0.276])
    if args.use_qmc_da:
        qmc_transforms, qmc_quotas = get_transforms(args)
        cifar100 = QMCDataset(dataset=datasets.CIFAR100(root=data_path, train=True, download=True),
            transforms=qmc_transforms, qmc_quotas=qmc_quotas, args=args)
        loaders['train'] = torch.utils.data.DataLoader(
            cifar100,
            batch_size=args.batch_size, shuffle=shuffle_flag,
            persistent_workers=False,
            num_workers=args.workers, pin_memory=False)
    else:
        if args.use_uniform_da:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        loaders['train'] = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_path, train=True, transform=transform_train, download=True),
            batch_size=args.batch_size, shuffle=shuffle_flag,
            num_workers=args.workers, pin_memory=False)

    loaders['val'] = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return loaders
    
def _get_mnist_loaders(args, data_path, shuffle_flag):
    loaders = {}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    loaders['train'] = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=shuffle_flag,
        num_workers=args.workers, pin_memory=False)
    loaders['val'] = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_path, train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return loaders

def _get_imagenet_loaders(args, shuffle_flag):
    loaders = {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    # use a sobol sequence that does sample index aware augmentation
    if args.use_qmc_da:
        qmc_transforms, qmc_quotas = get_transforms(args)
        imagenet = QMCDataset(dataset=ImageFolderLMDB(traindir),
            transforms=qmc_transforms, qmc_quotas=qmc_quotas, args=args)
        # loaders['trainset'] = cifar10
        loaders['train'] = torch.utils.data.DataLoader(
            imagenet,
            batch_size=args.batch_size, shuffle=shuffle_flag,
            persistent_workers=False,
            num_workers=args.workers, pin_memory=False)
    else:
        # use the original data augmentation
        if args.use_uniform_da:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        # no data augmentation in the training set
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        loaders['train'] = torch.utils.data.DataLoader(
            ImageFolderLMDB(traindir, transform_train),
            batch_size=args.batch_size, shuffle=shuffle_flag,
            num_workers=args.workers, pin_memory=False)

    loaders['val'] = torch.utils.data.DataLoader(
        ImageFolderLMDB(
            valdir,
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return loaders

def get_loaders(args):
    shuffle_flag = True if args.shuffle_type in [_RANDOM_RESHUFFLING_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_] else False
    data_path = os.path.join(args.data_path, "data")
    if args.dataset == _CIFAR10_:
        loaders = _get_cifar10_loaders(args, data_path, shuffle_flag)
    elif args.dataset == _CIFAR100_:
        loaders = _get_cifar100_loaders(args, data_path, shuffle_flag)    
    elif args.dataset == _MNIST_:
        loaders = _get_mnist_loaders(args, data_path, shuffle_flag)
    elif args.dataset == _IMAGENET_:
        loaders = _get_imagenet_loaders(args, shuffle_flag)
    else:
        raise NotImplementedError("This dataset is not supported, please choose from cifar10/cifar100, mnist.")
    return loaders


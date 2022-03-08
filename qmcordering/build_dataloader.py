# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .qmcda.datasets import CIFAR10, CIFAR100
from .qmcda.qmc_transforms import QMCCrop, QMCHorizontalFlip, QMCRotation, QuasiCIFAR10, QuasiCIFAR100, SobolNaiveQuasiCIFAR10, SobolCorrQuasiCIFAR10


"""
    The baseline data augmentation is taken from the following open source repos:
    cifar10: https://github.com/akamaster/pytorch_resnet_cifar10
    cifar100: https://github.com/weiaicunzai/pytorch-cifar100
    Currently, only CIFAR10, CIFAR100, and MNIST are supported
"""

def get_loaders(args):
    loaders = {}
    shuffle_flag = True if args.shuffle_type in ['RR', 'ZO'] else False
    if args.dataset == 'cifar10':
        data_path = os.path.join(args.data_path, "data")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # use a sobol sequence that does sample index aware augmentation
        if args.use_qmc_da:
            cifar10 = CIFAR10(cifar10=datasets.CIFAR10(root=data_path, train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['trainset'] = cifar10
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

    elif args.dataset == 'cifar100':
        data_path = os.path.join(args.data_path, "data")
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.268, 0.257, 0.276])
        if args.use_qmc_da:
            cifar100 = CIFAR100(cifar100=datasets.CIFAR100(root=data_path, train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['trainset'] = cifar100
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
    
    # TODO: The MNIST does not use the data augmentation, maybe we should use other variants?

    elif args.dataset == 'mnist':
        data_path = os.path.join(args.data_path, "data")
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
    else:
        raise NotImplementedError
    return loaders


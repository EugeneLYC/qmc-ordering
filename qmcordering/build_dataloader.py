# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
        
        # use a sobol sequence for loading the dataset (not RR or SO)
        if args.use_sobol_naive:
            qmccifar10 = SobolNaiveQuasiCIFAR10(cifar10=datasets.CIFAR10(root=data_path, train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['train'] = torch.utils.data.DataLoader(
                qmccifar10,
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)
        
        # use a sobol sequence that correlates the data augmentation
        elif args.use_sobol_corr:
            qmccifar10 = SobolCorrQuasiCIFAR10(cifar10=datasets.CIFAR10(root=data_path, train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['train'] = torch.utils.data.DataLoader(
                qmccifar10,
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)
        
        # use a sobol sequence that does sample index aware augmentation
        elif args.use_sample_aware_transform:
            qmccifar10 = QuasiCIFAR10(cifar10=datasets.CIFAR10(root=data_path, train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['train'] = torch.utils.data.DataLoader(
                qmccifar10,
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)
        
        # other cases where sobol sequence is not used
        else:
            # use the original data augmentation
            if args.use_uniform_da:
                transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ])
            
            # use the random thinning for qmc sequence
            elif args.use_qmc_da:
                transform_train = transforms.Compose([
                    QMCHorizontalFlip(),
                    QMCCrop(32, 4),
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
                num_workers=args.workers, pin_memory=True)

        loaders['val'] = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # elif args.dataset == 'cifar100':
    #     normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
    #                                  std=[0.268, 0.257, 0.276])
    #     if args.use_sample_aware_transform:
    #         qmccifar100 = QuasiCIFAR100(cifar100=datasets.CIFAR100(root=data_path, train=True, download=True),
    #             transform=transforms.Compose([
    #                 transforms.ToTensor(),
    #                 normalize,
    #             ]), args=args)
    #         loaders['train'] = torch.utils.data.DataLoader(
    #             qmccifar100,
    #             batch_size=args.batch_size, shuffle=shuffle_flag,
    #             num_workers=args.workers, pin_memory=True)
    #     else:
    #         if args.use_uniform_da:
    #             transform_train = transforms.Compose([
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.RandomCrop(32, 4),
    #                 transforms.RandomRotation(15),
    #                 transforms.ToTensor(),
    #                 normalize,
    #             ])
    #         elif args.use_qmc_da:
    #             transform_train = transforms.Compose([
    #                 QMCHorizontalFlip(),
    #                 QMCCrop(32, 4),
    #                 QMCRotation(15),
    #                 transforms.ToTensor(),
    #                 normalize,
    #             ])
    #         else:
    #             transform_train = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 normalize,
    #             ])
    #         loaders['train'] = torch.utils.data.DataLoader(
    #             datasets.CIFAR100(root=data_path, train=True, transform=transform_train, download=True),
    #             batch_size=args.batch_size, shuffle=shuffle_flag,
    #             num_workers=args.workers, pin_memory=True)

    #     loaders['val'] = torch.utils.data.DataLoader(
    #         datasets.CIFAR100(root=data_path, train=False, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             normalize,
    #         ])),
    #         batch_size=args.test_batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    
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
            num_workers=args.workers, pin_memory=True)

        loaders['val'] = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_path, train=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError
    return loaders


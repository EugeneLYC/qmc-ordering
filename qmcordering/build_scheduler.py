# -*- coding: utf-8 -*-
import torch

def get_lr_scheduler(args, optimizer):
    if 'resnet' in args.model:
        if args.dataset == 'cifar10':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch-1)
        elif args.dataset == 'cifar100':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120, 150], gamma=0.2, last_epoch=args.start_epoch-1)
        else:
            raise NotImplementedError
    elif args.model == 'logistic_regression' or args.model == 'lenet':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=args.start_epoch-1)
    else:
        raise NotImplementedError
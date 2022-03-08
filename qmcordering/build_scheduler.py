# -*- coding: utf-8 -*-
import torch
from .constants import *

def get_lr_scheduler(args, optimizer):
    if _RESNET_ in args.model:
        if args.dataset == _CIFAR10_:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch-1)
        elif args.dataset == _CIFAR100_:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120, 150], gamma=0.2, last_epoch=args.start_epoch-1)
        else:
            raise NotImplementedError
    elif args.model == _LOGISTIC_REGRESSION_ or args.model == _LENET_:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=args.start_epoch-1)
    else:
        raise NotImplementedError
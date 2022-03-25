# -*- coding: utf-8 -*-
import torch
from .constants import _LOGISTIC_REGRESSION_, _LENET_, _RESNET20_, _RESNET18_

def get_criterion(args):
    if args.model == _LOGISTIC_REGRESSION_:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == _LENET_:
        criterion = torch.nn.functional.nll_loss()
    elif args.model == _RESNET20_:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == _RESNET18_:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Unrecognized task")
    if torch.cuda.is_available():
        criterion.cuda()
    return criterion

def get_optimizer(args, model):
    return torch.optim.SGD(params=model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
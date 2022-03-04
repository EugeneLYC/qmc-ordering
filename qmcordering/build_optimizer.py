# -*- coding: utf-8 -*-
import torch

def get_criterion(args):
    if args.model == 'logistic_regression':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == 'lenet':
        criterion = torch.nn.functional.nll_loss()
    elif 'resnet' in args.model:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Unrecognized task")
    if torch.cuda.is_available():
        criterion.cuda()
    return criterion

def get_optimizer(args, model):
    # fuse the tensor
    # params = [
    #     {
    #         "params": [value],
    #         "name": key,
    #         "param_size": value.size(),
    #         "nelement": value.nelement(),
    #     }
    #     for key, value in model.named_parameters()
    # ]
    return torch.optim.SGD(params=model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
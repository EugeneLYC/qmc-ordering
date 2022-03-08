import torch
from .constants import *

def get_model(args):
    if args.model == _LOGISTIC_REGRESSION_:
        if args.dataset == _MNIST_:
            model = torch.nn.DataParallel(torch.nn.Linear(784, 10))
        else:
            raise NotImplementedError("Currently only MNIST is supported for this model")
    elif args.model == _LENET_:
        from .models import LeNet
        model = torch.nn.DataParallel(LeNet())
    elif args.model == _RESNET20_:
        if args.dataset == _CIFAR10_:
            from .models import resnet
            model = torch.nn.DataParallel(resnet.__dict__[args.model]())
        else:
            raise NotImplementedError("Currently only CIFAR10 is supported for this model")
    else:
        raise NotImplementedError("This model is currently not supported, please add its implementation in qmcordering/models")
    if torch.cuda.is_available():
        model.cuda()
    dimension = report_model_info(model)
    return model, dimension

def report_model_info(model):
    dimension = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Creating model with dimension: {}.".format(
            dimension,
        )
    )
    return dimension
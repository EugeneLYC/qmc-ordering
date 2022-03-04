import torch
from .models import LeNet, resnet

def get_model(args):
    if args.model == 'logistic_regression':
        if args.dataset == 'mnist':
            model = torch.nn.DataParallel(torch.nn.Linear(784, 10))
        else:
            raise NotImplementedError("Currently only MNIST is supported for this model")
    elif args.model == 'lenet':
        model = torch.nn.DataParallel(LeNet())
    elif args.model == 'resnet20':
        if args.dataset == 'cifar10':
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
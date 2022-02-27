import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from qmc_transforms import QMCCrop, QMCHorizontalFlip, QMCRotation, QuasiCIFAR10, QuasiCIFAR100, SobolNaiveQuasiCIFAR10, SobolCorrQuasiCIFAR10
import torchvision.datasets as datasets
import resnet
from lenet import LeNet
from tensorboardX import SummaryWriter
from utils import *
from timer import Timer

parser = argparse.ArgumentParser(description='Experimental code for the QMC paper')
parser.add_argument('--model', metavar='ARCH', default='resnet20',
                    help='model to use (lenet, resnetxx)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset used in the experiment (default: cifar10)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--test_batch_size', default=1024, type=int,
                    metavar='N', help='mini-batch size used for testing (default: 1024)')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--start_greedy', default=1, type=int,
                    metavar='N', help='the epoch where the greedy strategy will be first used (100 in CIFAR10 case)')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save_every', dest='save_every', type=int, default=10,
                    help='Saves checkpoints at every specified number of epochs')
parser.add_argument('--input_size', default=784, type=int, metavar='N',
                    help='input size for MNIST dataset (28*28=784 by default)')
parser.add_argument('--num_classes', default=10, type=int, metavar='N',
                    help='number of classes for the MNIST dataset (10 classes in total)')
parser.add_argument('--seed', default=0, type=int, metavar='N',
                    help='random seed used in the experiment')
parser.add_argument('--log_tune_seeds', default=False, action='store_true',
                    help='log the seeds results in a txt file for consistent results')
parser.add_argument('--use_tensorboard', default=False, action='store_true',
                    help='log the seeds results in a txt file for consistent results')

# greedy method related arguments
parser.add_argument('--shuffle_type', default='RR', choices=['RR', 'SO', 'greedy'], type=str,
                    help='shuffle type used for the optimization (choose from RR, SO, greedy)')
parser.add_argument('--task_name', default='test', type=str,
                    help='task name used for tensorboard')
parser.add_argument('--log_metric', default=False, action='store_true',
                    help='whether to log the LHS-QMC metric during training (default: False)')
parser.add_argument('--use_random_proj', default=False, action='store_true',
                    help='whether to use projection when doing the greedy sorting (default: True)')
parser.add_argument('--use_random_proj_full', default=False, action='store_true',
                    help='whether to use projection after storing all the full-dimension gradients (default: True)')
parser.add_argument('--use_qr', default=False, action='store_true',
                    help='whether to use qr_decomposition in the sorting part (default: True)')
parser.add_argument('--proj_ratio', default=0.1, type=float,
                    help='decide project how much ratio of the orginal entire model (default: 0.1)')

# data augmentation related arguments
parser.add_argument('--use_uniform_da', default=False, action='store_true',
                    help='whether to use the baseline data augmentation in the training data')
parser.add_argument('--use_qmc_da', default=False, action='store_true',
                    help='whether to use qmc based data augmentation (now for cifar)')
parser.add_argument('--use_sample_aware_transform', default=False, action='store_true',
                    help='whether to use sample aware transform in the dataset loading')
parser.add_argument('--use_sobol_naive', default=False, action='store_true',
                    help='whether to use sample aware transform in the dataset loading')
parser.add_argument('--use_sobol_corr', default=False, action='store_true',
                    help='whether to use sample aware transform in the dataset loading')

# sobol sequence related argument
parser.add_argument('--sobol_type', default='overlap', choices=['independent', 'overlap', 'identical'], type=str,
                    help='sobol_type used for generating low-discrepency sequence (choose from independent, overlap, identical)')

parser.add_argument('--use_self_timer', default=True, action='store_true',
                    help='whether to use self-written timer to time different parts (default: True)')

args = parser.parse_args()

def get_model(args):
    
    # TODO: Fix the resnet loading: combine the resnet20 with others

    if args.model == 'logistic_regression':
        model = torch.nn.DataParallel(nn.Linear(args.input_size, args.num_classes))
    elif args.model == 'lenet':
        model = torch.nn.DataParallel(LeNet())
    elif args.model == 'resnet20':
        if args.dataset == 'cifar10':
            model = torch.nn.DataParallel(resnet.__dict__[args.model]())
        else:
            from models.resnet import ResNet_cifar
            model = torch.nn.DataParallel(ResNet_cifar(dataset='cifar100', resnet_size=20))
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        model.cuda()
    return model

"""
    The baseline data augmentation is taken from the following open source repos:
    cifar10: https://github.com/akamaster/pytorch_resnet_cifar10
    cifar100: https://github.com/weiaicunzai/pytorch-cifar100
    Currently, only CIFAR10, CIFAR100, and MNIST are supported
"""

def get_loaders(args):
    loaders = {}
    shuffle_flag = True if args.shuffle_type == 'RR' else False
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # use a sobol sequence for loading the dataset (not RR or SO)
        if args.use_sobol_naive:
            qmccifar10 = SobolNaiveQuasiCIFAR10(cifar10=datasets.CIFAR10(root='./data', train=True, download=True),
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
            qmccifar10 = SobolCorrQuasiCIFAR10(cifar10=datasets.CIFAR10(root='./data', train=True, download=True),
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
            qmccifar10 = QuasiCIFAR10(cifar10=datasets.CIFAR10(root='./data', train=True, download=True),
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
                datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True),
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)

        loaders['val'] = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.268, 0.257, 0.276])
        if args.use_sample_aware_transform:
            qmccifar100 = QuasiCIFAR100(cifar100=datasets.CIFAR100(root='./data', train=True, download=True),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args=args)
            loaders['train'] = torch.utils.data.DataLoader(
                qmccifar100,
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)
        else:
            if args.use_uniform_da:
                transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif args.use_qmc_da:
                transform_train = transforms.Compose([
                    QMCHorizontalFlip(),
                    QMCCrop(32, 4),
                    QMCRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            loaders['train'] = torch.utils.data.DataLoader(
                datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True),
                batch_size=args.batch_size, shuffle=shuffle_flag,
                num_workers=args.workers, pin_memory=True)

        loaders['val'] = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    # TODO: The MNIST does not use the data augmentation, maybe we should use other variants?

    elif args.dataset == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        loaders['train'] = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=shuffle_flag,
            num_workers=args.workers, pin_memory=True)

        loaders['val'] = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError
    return loaders

def get_criterion(args):
    if args.model == 'logistic_regression':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'lenet':
        criterion = F.nll_loss()
    elif 'resnet' in args.model:
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        criterion.cuda()
    return criterion

def get_optimizer(args, model):
    # fuse the tensor
    params = [
        {
            "params": [value],
            "name": key,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]
    return torch.optim.SGD(params=params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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

def _build_task_name(args):
    task_name = 'MODEL-' + args.model + \
                '_DATA-' + args.dataset + \
                '_STYPE-' + args.shuffle_type + \
                '_SEED-' + str(args.seed) + \
                '_SOBOL-' + args.sobol_type
    if args.use_qmc_da or args.use_sample_aware_transform:
        task_name = 'QMCDA_' + task_name
    else:
        task_name = 'DA_' + task_name
    return task_name

def main():
    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model        = get_model(args)
    loaders      = get_loaders(args)
    criterion    = get_criterion(args)
    optimizer    = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    args.task_name = _build_task_name(args)

    timer = Timer(verbosity_level=1, on_cuda=True)
    if args.use_tensorboard:
        logger = SummaryWriter('./runs/' + args.task_name)
    else:
        logger = None

    tr_acc_w, tr_loss_w, te_acc_w, te_loss_w = None, None, None, None
    if args.log_tune_seeds:
        if args.shuffle_type == 'greedy':
            tr_loss_w = open('./txt_greedy/tr_loss_greedy_qr_' + str(args.seed), 'a')
            tr_acc_w = open('./txt_greedy/tr_acc_greedy_qr_' + str(args.seed), 'a')
            te_loss_w = open('./txt_greedy/te_loss_greedy_qr_' + str(args.seed), 'a')
            te_acc_w = open('./txt_greedy/te_acc_greedy_qr_' + str(args.seed), 'a')
        else:
            tr_loss_w = open('./txt_greedy/tr_loss_'+args.shuffle_type+'_' + str(args.seed), 'a')
            tr_acc_w = open('./txt_greedy/tr_acc_'+args.shuffle_type+'_' + str(args.seed), 'a')
            te_loss_w = open('./txt_greedy/te_loss_'+args.shuffle_type+'_' + str(args.seed), 'a')
            te_acc_w = open('./txt_greedy/te_acc_'+args.shuffle_type+'_' + str(args.seed), 'a')

    if args.shuffle_type == 'greedy':
        intermediate_results = {
            'param_names':list(enumerate([group['name'] for group in optimizer.param_groups]))
        }

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        if args.shuffle_type == 'greedy':
            intermediate_results = train_greedy(args, loaders['train'], model, criterion, optimizer, epoch, logger, timer, intermediate_results, tr_acc_w, tr_loss_w)
        elif args.shuffle_type == 'RR' or args.shuffle_type == 'SO':
            train(args, loaders['train'], model, criterion, optimizer, epoch, logger, timer, tr_acc_w, tr_loss_w)
        else:
            raise NotImplementedError
        lr_scheduler.step()

        # evaluate on validation set
        validate(args, loaders['val'], model, criterion, epoch, logger, te_acc_w, te_loss_w)

        print(timer.summary())
    logger.close()

    if args.log_tune_seeds:
        tr_loss_w.close()
        tr_acc_w.close()
        te_loss_w.close()
        te_acc_w.close()
    
    print("finish training")

if __name__ == '__main__':
    main()

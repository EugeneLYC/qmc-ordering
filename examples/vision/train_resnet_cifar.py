import os
import random
import torch
import logging
from tensorboardX import SummaryWriter

from models import resnet
from arguments import get_args
from utils import _load_batch, _inference, _backward, train, validate, Timer

from qmcorder.qmcda.datasets import Dataset
from constants import _RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_, _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_

logger = logging.getLogger(__name__)

def main():
    args = get_args()

    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Using random seed {args.seed} for random and torch module.")

    args.use_cuda = torch.cuda.is_available()
    logger.info(f"Using GPU: {args.use_cuda}")

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    model = torch.nn.DataParallel(resnet.__dict__[args.model]())
    if args.use_cuda:
        model.cuda()
    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Using model: {args.model} with dimension: {model_dimen}.")

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda():
        criterion.cuda()
    logger.info(f"Using Cross Entropy Loss for classification.")

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logger.info(f"Using optimizer SGD with hyperparameters: learning rate={args.lr}; momentum={args.momentum}; weight decay={args.weight_decay}.")

    if args.dataset == _CIFAR10_:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150],
                                                            last_epoch=args.start_epoch-1)
    elif args.dataset == _CIFAR100_:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[60, 120, 150],
                                                            gamma=0.2,
                                                            last_epoch=args.start_epoch-1)
    else:
        raise NotImplementedError("This script is for CIFAR datasets. Please input cifar10 or cifar100 in --dataset.")
    logger.info(f"Using dataset: {args.dataset}")

    # Below is the code that reflects our change to the training pipeline.
    # In the dataloader, we can use the loader with the QMC-based data augmentation enabled.
    # We can additionally add a sorter for epoch-wise example ordering.
    # For more technical details, please refer to https://openreview.net/pdf?id=7gWSJrP3opB.

    # QMC-based data augmentation

    loaders = {}
    if args.dataset == _CIFAR10_:
        trainset = Dataset(dataset=datasets.CIFAR10(root=data_path, train=True, download=True),
                            train=True,
                            args=args)
        testset = Dataset(dataset=datasets.CIFAR10(root=data_path, train=False),
                            train=False,
                            args=args)
    elif args.dataset == _CIFAR100_:
        trainset = Dataset(dataset=datasets.CIFAR100(root=data_path, train=True, download=True),
                            train=True,
                            args=args)
        testset = Dataset(dataset=datasets.CIFAR100(root=data_path, train=False),
                            train=False,
                            args=args)
    else:
        raise NotImplementedError("This script is for CIFAR datasets. Please input cifar10 or cifar100 in --dataset.")
    loaders['train'] = torch.utils.data.DataLoader(trainset,
                                                    batch_size=args.batch_size,
                                                    shuffle=shuffle_flag,
                                                    persistent_workers=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False)
    loaders['val'] = torch.utils.data.DataLoader(testset,
                                                    batch_size=args.test_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False)
    
    # Epoch-wise data ordering
    if args.shuffle_type in [_RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_]:
        sorter = None
        logger.info(f"Not using any sorting algorithm.")
    else:
        logger.info(f"Creating sorting algorithm: {args.shuffle_type}.")
        num_batches = len(list(enumerate(loader)))
        if args.shuffle_type == _STALE_GRAD_SORT_:
            from qmcorder.sort.algo import StaleGradGreedySort
            sorter = StaleGradGreedySort(args,
                                        num_batches,
                                        grad_dimen=dimension,
                                        timer=timer)
        elif args.shuffle_type == _ZEROTH_ORDER_SORT_:
            from qmcorder.sort.algo import ZerothOrderGreedySort
            sorter = ZerothOrderGreedySort(args,
                                        num_batches,
                                        grad_dimen=dimension,
                                        model=model,
                                        timer=timer)
        elif args.shuffle_type == _FRESH_GRAD_SORT_:
            from qmcorder.sort.algo import FreshGradGreedySort
            sorter = FreshGradGreedySort(args,
                                        num_batches,
                                        grad_dimen=dimension,
                                        timer=timer)
        else:
            raise NotImplementedError("This sorting method is not supported yet")

    args.task_name = build_task_name(args)
    logger.info(f"Creating task name as: {args.task_name}.")

    if args.use_tensorboard:
        tb_path = os.path.join(args.tensorboard_path, 'runs', args.task_name)
        logger.info(f"Streaming tensorboard logs to path: {tb_path}.")
        tb_logger = SummaryWriter(tb_path)
    else:
        tb_logger = None
        logger.info(f"Disable tensorboard logs currently.")

    for epoch in range(args.start_epoch, args.epochs):
        train(args=rgs,
            loader=loaders['train'],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=tb_logger,
            timer=timer,
            sorter=sorter)
        
        lr_scheduler.step()

        # evaluate on validation set
        validate(args=args,
                loader=loaders['val'],
                model=model,
                criterion=criterion,
                epoch=epoch,
                tb_logger=tb_logger)

    tb_logger.close()

    logger.info(f"Finish training!")

if __name__ == '__main__':
    main()

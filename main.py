import os
import random
import torch
from tensorboardX import SummaryWriter
from qmcordering.utils import train, validate, Timer
from qmcordering import build_model, build_dataloader, build_optimizer, build_scheduler, build_sorter
from arguments import get_args

def _build_task_name(args):
    task_name = 'MODEL-' + args.model + \
                '_DATA-' + args.dataset + \
                '_SFTYPE-' + args.shuffle_type + \
                '_SEED-' + str(args.seed)
    task_name = task_name + '_lr-' + str(args.lr)
    if args.use_qmc_da:
        task_name = 'QMCDA' + task_name
    if args.shuffle_type == 'ZO':
        task_name = task_name + '_ZOBSZ-' + str(args.zo_batch_size)
    if args.shuffle_type == 'fresh':
        task_name = task_name + '_proj-' + str(args.zo_batch_size)
    if args.shuffle_type == 'greedy' and args.use_random_proj:
        task_name = task_name + '_proj-' + str(args.proj_target)
    return task_name

def main():
    args = get_args()
    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.use_cuda = torch.cuda.is_available()

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    model, dimension = build_model.get_model(args)
    loaders = build_dataloader.get_loaders(args)
    criterion = build_optimizer.get_criterion(args)
    optimizer = build_optimizer.get_optimizer(args, model)
    lr_scheduler = build_scheduler.get_lr_scheduler(args, optimizer)
    if args.shuffle_type in ['greedy', 'fresh']:
        sorter = build_sorter.get_sorter(args,
                                        loaders['train'],
                                        dimension,
                                        timer=timer)
    elif args.shuffle_type in ['ZO']:
        sorter = build_sorter.get_sorter(args,
                                        loaders['train'],
                                        dimension,
                                        model=model,
                                        timer=timer)
    else:
        sorter = None

    args.task_name = _build_task_name(args)

    if args.use_tensorboard:
        tb_path = os.path.join(args.tensorboard_path, 'runs', args.task_name)
        logger = SummaryWriter(tb_path)
    else:
        logger = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.use_qmc_da:
            loaders['trainset'].update(epoch)
        train(args,
            loaders['train'],
            model,
            criterion,
            optimizer,
            epoch,
            logger,
            timer=timer,
            sorter=sorter)
        lr_scheduler.step()

        # evaluate on validation set
        validate(args,
                loaders['val'],
                model,
                criterion,
                epoch,
                logger)

    logger.close()

    print("finish training")

if __name__ == '__main__':
    main()

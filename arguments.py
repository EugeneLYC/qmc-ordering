import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Experimental code for the QMC paper')

    parser.add_argument('--model',
                        metavar='ARCH',
                        default='resnet20',
                        help='model to use (lenet, resnetxx)')

    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='dataset used in the experiment (default: cifar10)')
    
    parser.add_argument('--data_path',
                        type=str,
                        help='the base directory for dataset')
    
    parser.add_argument('--workers',
                        default=0,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 0)')
    
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')
    
    parser.add_argument('--test_batch_size',
                        default=1024,
                        type=int,
                        metavar='N',
                        help='mini-batch size used for testing (default: 1024)')
    
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    
    parser.add_argument('--weight_decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')
    
    parser.add_argument('--print_freq',
                        default=50,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 50)')
    
    parser.add_argument('--start_greedy',
                        default=1,
                        type=int,
                        metavar='N',
                        help='the epoch where the greedy strategy will be first used (100 in CIFAR10 case)')
    
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        metavar='N',
                        help='random seed used in the experiment')
    
    parser.add_argument('--log_tune_seeds',
                        default=False,
                        action='store_true',
                        help='log the seeds results in a txt file for consistent results')
    
    parser.add_argument('--use_tensorboard',
                        default=False,
                        action='store_true',
                        help='log the seeds results in a txt file for consistent results')
    
    parser.add_argument('--tensorboard_path',
                        type=str,
                        help='the base directory for tensorboard logs')
    
    parser.add_argument('--zo_batch_size',
                        default=1,
                        type=int,
                        metavar='N',
                        help='zero-th order mini-batch size (default: 16)')

    # greedy method related arguments
    parser.add_argument('--shuffle_type',
                        default='RR',
                        choices=['RR', 'SO', 'greedy', 'ZO'],
                        type=str,
                        help='shuffle type used for the optimization (choose from RR, SO, greedy)')
    
    parser.add_argument('--task_name',
                        default='test',
                        type=str,
                        help='task name used for tensorboard')
    
    parser.add_argument('--log_metric',
                        default=False,
                        action='store_true',
                        help='whether to log the LHS-QMC metric during training (default: False)')
    
    parser.add_argument('--use_random_proj',
                        default=False,
                        action='store_true',
                        help='whether to use projection when doing the greedy sorting (default: True)')
    
    parser.add_argument('--use_random_proj_full',
                        default=False,
                        action='store_true',
                        help='whether to use projection after storing all the full-dimension gradients (default: True)')
    
    parser.add_argument('--use_qr',
                        default=False,
                        action='store_true',
                        help='whether to use qr_decomposition in the sorting part (default: True)')
    
    parser.add_argument('--proj_ratio',
                        default=0.1,
                        type=float,
                        help='decide project how much ratio of the orginal entire model (default: 0.1)')

    # data augmentation related arguments
    parser.add_argument('--use_uniform_da',
                        default=False,
                        action='store_true',
                        help='whether to use the baseline data augmentation in the training data')
    
    parser.add_argument('--use_qmc_da',
                        default=False,
                        action='store_true',
                        help='whether to use qmc based data augmentation (now for cifar)')
    
    parser.add_argument('--use_sample_aware_transform',
                        default=False,
                        action='store_true',
                        help='whether to use sample aware transform in the dataset loading')
    
    parser.add_argument('--use_sobol_naive',
                        default=False,
                        action='store_true',
                        help='whether to use sample aware transform in the dataset loading')
    
    parser.add_argument('--use_sobol_corr',
                        default=False,
                        action='store_true',
                        help='whether to use sample aware transform in the dataset loading')

    # sobol sequence related argument
    parser.add_argument('--sobol_type',
                        default='overlap',
                        choices=['independent', 'overlap', 'identical'],
                        type=str,
                        help='sobol_type used for generating low-discrepency sequence (choose from independent, overlap, identical)')

    parser.add_argument('--use_self_timer',
                        default=True,
                        action='store_true',
                        help='whether to use self-written timer to time different parts (default: True)')

    args = parser.parse_args()
    return args

import torch
import numpy as np
import random
from fusetensor import *
from sklearn import random_projection
from timer import Timer

def _load_batch(args, input, target):
    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()
    if args.dataset == 'mnist':
        input = input.reshape(-1, args.input_size)
    return input, target

def random_proj(data):
    rp = random_projection.SparseRandomProjection(random_state=1)
    return torch.from_numpy(rp.fit_transform(data))

def _compute_qmc_metric(args,
                        model,
                        train_batches,
                        criterion,
                        optimizer,
                        epoch,
                        logger,
                        orders=None):
    grads = dict()
    for i in range(len(train_batches)):
        grads[i] = [p.data.clone().zero_() for p in model.parameters()]
    full_grad = [p.data.clone().zero_() for p in model.parameters()]
    if orders is None:
        orders = {i:0 for i in range(len(train_batches))}
    for j in orders.keys():
        i, (input, target) = train_batches[j]
        input_var, target_var = _load_batch(args, input, target)
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        for m, p in enumerate(model.parameters()):
            grads[i][m] = p.grad.data.clone()
            full_grad[m].add_(p.grad.data.clone())
    cur_grad = [p.data.clone().zero_() for p in model.parameters()]
    cur_var = 0
    index=0
    for j in orders.keys():
        i, _ = train_batches[j]
        for p1, p2, p3 in zip(cur_grad, grads[i], full_grad):
            p1.data.add_(p2.data)
            cur_var += torch.norm(p1.data/(index+1) - p3.data/len(train_batches)).item()**2
        index += 1
    logger.add_scalar('train/metric', cur_var, epoch)

def train(args,
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        logger,
        timer,
        acc_w=None,
        loss_w=None):
    if args.log_metric:
        train_batches = list(enumerate(train_loader))
        _compute_qmc_metric(args, model, train_batches, criterion, optimizer, epoch, logger)
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):

        with timer("load batch", epoch=epoch):
            input_var, target_var = _load_batch(args, input, target)
        
        with timer("forward pass", epoch=epoch):
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
        
        with timer("backward pass", epoch=epoch):
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))
    if args.use_tensorboard:
        logger.add_scalar('train/accuracy', top1.avg, epoch)
        logger.add_scalar('train/loss', losses.avg, epoch)
    if args.log_tune_seeds:
        acc_w.write(str(epoch)+'\t'+str(top1.avg)+'\n')
        loss_w.write(str(epoch)+'\t'+str(losses.avg)+'\n')


def train_greedy(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                logger,
                timer,
                intermediate_results=None,
                acc_w=None,
                loss_w=None):
    """
        Run one train epoch
    """
    
    # the sorting part of the greedy method
    
    train_batches = list(enumerate(train_loader))
    orders = {i:0 for i in range(len(train_batches))}
    if epoch > args.start_greedy:
        cur_sum = intermediate_results['fused_full_grad'].clone().zero_()
        remain_ids = set(range(len(train_batches)))
        if args.use_qr or args.use_random_proj_full:
            with timer("dimension reduction auxiliary", epoch=epoch):
                X = torch.zeros(len(train_batches)+1, len(intermediate_results['fused_grads'][0]))
                for i in range(len(train_batches)):
                    X[i] = intermediate_results['fused_grads'][i]
                X[len(train_batches)] = intermediate_results['fused_full_grad']
            if args.use_qr:
                with timer("QR", epoch=epoch):
                    _, X = torch.qr(X.t())
                    X = X.t()
            else:
                with timer("random projection as full matrix", epoch=epoch):
                    # now the random projection is implemented using sklearn library
                    # cuda operations are not supported
                    X = X.cpu()
                    rp = random_projection.SparseRandomProjection()
                    X = torch.from_numpy(rp.fit_transform(X)).cuda()
            with timer("dimension reduction auxiliary", epoch=epoch):
                cur_sum = X[0].clone().zero_()
                for i in range(len(train_batches)):
                    intermediate_results['fused_grads'][i] = X[i]
                intermediate_results['fused_full_grad'] = X[len(train_batches)]
        with timer("greedy sorting", epoch=epoch):
            for i in range(1, len(train_batches)+1):
                cur_id = -1
                max_norm = float('inf')
                for candidate_id in remain_ids:
                    candidate_norm = 0
                    candidate_norm = torch.norm(
                        (intermediate_results['fused_grads'][candidate_id] + cur_sum*(i-1))/i - intermediate_results['fused_full_grad']/len(train_batches)
                    ).item()
                    if candidate_norm < max_norm:
                        max_norm = candidate_norm
                        cur_id = candidate_id
                remain_ids.remove(cur_id)
                orders[cur_id] = i
                cur_sum = (cur_sum*(i-1) + intermediate_results['fused_grads'][cur_id].clone())/i
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}

    if args.log_metric:
        _compute_qmc_metric(args,
                            model,
                            train_batches,
                            criterion,
                            optimizer,
                            epoch,
                            logger,
                            orders=orders)
    
    losses = AverageMeter()
    top1 = AverageMeter()

    with timer("fuse gradients", epoch=epoch):
        fused_grads = dict()
        fused_full_grad = None
        params, _ = get_data(
            optimizer.param_groups, intermediate_results['param_names'], is_get_grad=False
        )
        temp_tensor = TensorBuffer(params).buffer.clone()
        d = len(temp_tensor)

    # switch to train mode

    model.train()
    # use the same random matrix throughout one epoch
    if args.use_random_proj:
        rs = random.randint(0, 1000)
        rp = random_projection.SparseRandomProjection(n_components=int(args.proj_ratio*d), random_state=rs)

    for i in orders.keys():
        _, (input, target) = train_batches[i]
        with timer("load batch", epoch=epoch):
            input_var, target_var = _load_batch(args, input, target)
        
        with timer("forward pass", epoch=epoch):
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        with timer("backward pass", epoch=epoch):
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

        if epoch >= args.start_greedy:
            with timer("store gradients", epoch=epoch):
                params, _ = get_data(
                    optimizer.param_groups, intermediate_results['param_names'], is_get_grad=True
                )
                gi = TensorBuffer(params).buffer.clone()
                if args.use_random_proj:
                    with timer("cpu-gpu transport", epoch=epoch):
                        gi = gi.cpu()
                    with timer("random projection", epoch=epoch):
                        gi = torch.from_numpy(rp.fit_transform(gi.reshape(1, -1)))
                    with timer("cpu-gpu transport", epoch=epoch):
                        gi = gi.cuda()
                    fused_grads[i] = gi[0]
                else:
                    fused_grads[i] = gi
                if fused_full_grad is None:
                    fused_full_grad = fused_grads[i].clone()
                else:
                    fused_full_grad.add_(fused_grads[i].clone())

        with timer("backward pass", epoch=epoch):
            optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))
    if args.use_tensorboard:
        logger.add_scalar('train/accuracy', top1.avg, epoch)
        logger.add_scalar('train/loss', losses.avg, epoch)
    if args.log_tune_seeds:
        acc_w.write(str(epoch)+'\t'+str(top1.avg)+'\n')
        loss_w.write(str(epoch)+'\t'+str(losses.avg)+'\n')

    """

    code for random sparsification, we do not need these anymore

    with timer("random sparsification", epoch=epoch):
        selected_coo_id = torch.randperm(fused_full_grad.shape[0])
        # to perform the random projection, first random permute the entire tensor, then get the first X percent of the tensor coordiantes
        for i in orders.keys():
            fused_grads[i] = fused_grads[i][selected_coo_id].view(fused_grads[i].size())[:max(int(len(selected_coo_id)*args.projection_ratio), 1)]
        fused_full_grad = fused_full_grad[selected_coo_id].view(fused_full_grad.size())[:max(int(len(selected_coo_id)*args.projection_ratio), 1)]
    """

    intermediate_results = {
        'fused_grads':fused_grads,
        'fused_full_grad':fused_full_grad,
        'param_names':intermediate_results['param_names']
    }
    return intermediate_results


def validate(args, val_loader, model, criterion, epoch, logger, acc_w=None, loss_w=None):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = _load_batch(args, input, target)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), loss=losses,
                          top1=top1))
    if args.use_tensorboard:
        logger.add_scalar('test/accuracy', top1.avg, epoch)
        logger.add_scalar('test/loss', losses.avg, epoch)
    if args.log_tune_seeds:
        acc_w.write(str(epoch)+'\t'+str(top1.avg)+'\n')
        loss_w.write(str(epoch)+'\t'+str(losses.avg)+'\n')

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
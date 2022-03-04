import torch
import time
import numpy as np
from contextlib import contextmanager
from io import StringIO
from .sort.utils import _load_batch, compute_avg_grad_error

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

def train(args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            logger,
            timer=None,
            sorter=None):
    train_batches = list(enumerate(train_loader))
    if sorter is not None:
        with timer("sorting", epoch=epoch):
            orders = sorter.sort(epoch)

    if args.log_metric:
        compute_avg_grad_error(args,
                            model,
                            train_batches,
                            criterion,
                            optimizer,
                            epoch,
                            logger,
                            orders=orders)
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode

    model.train()

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

        if sorter is not None and args.shuffle_type == 'greedy':
            with timer("sorting", epoch=epoch):
                sorter.update_stale_grad(optimizer, i, epoch)
        
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
        total_time = timer.totals["load batch"] + timer.totals["forward pass"] + \
            timer.totals["backward pass"] + timer.totals["sorting"]
        logger.add_scalar('train_time/accuracy', top1.avg, total_time)
        logger.add_scalar('train_time/loss', losses.avg, total_time)


    return



def validate(args, val_loader, model, criterion, epoch, logger):
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

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return

class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:
    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, skip_first=True, use_cuda=True):
        self.verbosity_level = verbosity_level
        #self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first
        self.cuda_available = torch.cuda.is_available() and use_cuda

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time()
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if label not in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if label not in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif label not in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        #if self.call_counts[label] > 0:
        #    # We will reduce the probability of logging a timing
        #    # linearly with the number of time we have seen it.
        #    # It will always be recorded in the totals, though.
        #    if np.random.rand() < 1 / self.call_counts[label]:
        #        self.log_fn(
        #            "timer", {"epoch": epoch, "value": end - start}, {"event": label}
        #        )

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        if len(self.totals) > 0:
            with StringIO() as buffer:
                total_avg_time = 0
                print("--- Timer summary ------------------------", file=buffer)
                print("  Event   |  Count | Average time |  Frac.", file=buffer)
                for event_label in sorted(self.totals):
                    total = self.totals[event_label]
                    count = self.call_counts[event_label]
                    if count == 0:
                        continue
                    avg_duration = total / count
                    total_runtime = (
                        self.last_time[event_label] - self.first_time[event_label]
                    )
                    runtime_percentage = 100 * total / total_runtime
                    total_avg_time += avg_duration if "." not in event_label else 0
                    print(
                        f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                        file=buffer,
                    )
                print("-------------------------------------------", file=buffer)
                event_label = "total_averaged_time"
                print(
                    f"- {event_label:30s}| {count:6d} | {total_avg_time:11.5f}s |",
                    file=buffer,
                )
                print("-------------------------------------------", file=buffer)
                return buffer.getvalue()

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if self.cuda_available:
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")

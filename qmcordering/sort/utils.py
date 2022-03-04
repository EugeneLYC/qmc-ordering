import torch
from sklearn import random_projection

def _load_batch(args, input, target):
    if args.use_cuda:
        input = input.cuda()
        target = target.cuda()
    if args.dataset == 'mnist':
        # additionally reshaping the mnist input for compatability
        input = input.reshape(-1, 784)
    return input, target

def random_proj(data):
    rp = random_projection.SparseRandomProjection(random_state=1)
    return torch.from_numpy(rp.fit_transform(data))

def compute_avg_grad_error(args,
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

def flatten_grad(optimizer):
    t = None
    for _, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            if p.grad.data is not None:
                if t is None:
                    t = torch.flatten(p.grad.data)
                else:
                    t = torch.cat(
                        (t, torch.flatten(p.grad.data))
                    )
    return t

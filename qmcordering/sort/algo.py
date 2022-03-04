import torch
import copy
import random
from sklearn import random_projection
from .utils import flatten_grad


class Sort:
    def sort(self, orders):
        raise NotImplementedError


class StaleGradGreedySort(Sort):
    def __init__(self,
                args,
                num_batches,
                grad_dimen,
                timer=None):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.timer = timer
        assert self.timer is not None
        self.stale_grad_matrix = torch.zeros(num_batches, grad_dimen)
        self.avg_grad = torch.zeros(grad_dimen)
        self._reset_random_proj_matrix()
        
    def report_progress(self):
        print(self.timer.summary())
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_greedy
    
    def _reset_random_proj_matrix(self):
        rs = random.randint(0, 10000)
        self.rp = random_projection.SparseRandomProjection(n_components=int(self.args.proj_ratio*self.grad_dimen), random_state=rs)
    
    def update_stale_grad(self, optimizer, batch_idx, epoch, add_to_avg=True):
        tensor = flatten_grad(optimizer)
        if self.args.use_random_proj:
            with self.timer("cpu-gpu transport", epoch=epoch):
                tensor = tensor.cpu()
            with self.timer("random projection", epoch=epoch):
                tensor = torch.from_numpy(self.rp.fit_transform(tensor.reshape(1, -1)))
            with self.timer("cpu-gpu transport", epoch=epoch):
                self.tensor = tensor.cuda()
            self.stale_grad_matrix[batch_idx].copy_(tensor[0])
        else:
            self.stale_grad_matrix[batch_idx].copy_(tensor)
        if add_to_avg:
            self.avg_grad.add_(tensor / self.num_batches)
        # make sure the same random matrix is used in one epoch
        if batch_idx == self.num_batches - 1 and self.args.use_random_proj:
            self._reset_random_proj_matrix()

    def sort(self, epoch, orders=None):
        if orders is None:
            orders = {i:0 for i in range(self.num_batches)}
        if self._skip_sort_this_epoch(epoch):
            return orders
        if self.args.use_qr:
            assert self.args.use_random_proj_full is False
            with self.timer("QR decomposition", epoch=epoch):
                _, X = torch.qr(self.stale_grad_matrix.t())
                X = X.t()
        if self.args.use_random_proj_full:
            with self.timer("random projection as full matrix", epoch=epoch):
                # Since the random projection is implemented using sklearn library,
                # cuda operations are not supported
                X = self.stale_grad_matrix.clone()
                if self.args.use_cuda:
                    X = X.cpu()
                rp = random_projection.SparseRandomProjection()
                X = torch.from_numpy(rp.fit_transform(X))
                if self.args.use_cuda:
                    X = X.cuda()
        if not (self.args.use_qr and self.args.use_random_proj_full):
            X = self.stale_grad_matrix.clone()
        cur_sum = torch.zeros_like(self.avg_grad)
        remain_ids = set(range(self.num_batches))
        with self.timer("greedy sorting", epoch=epoch):
            for i in range(1, self.num_batches+1):
                cur_id = -1
                max_norm = float('inf')
                for cand_id in remain_ids:
                    cand_norm = torch.norm(
                        (X[cand_id] + cur_sum*(i-1))/i - self.avg_grad
                    ).item()
                    if cand_norm < max_norm:
                        max_norm = cand_norm
                        cur_id = cand_id
                remain_ids.remove(cur_id)
                orders[cur_id] = i
                cur_sum.mul_(i-1).add_(X[cur_id]).mul_(1/i)
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders
        
    

class ZerothOrderGreedySort(Sort):
    def __init__(self, args, timing=True):
        pass

    def sort(self, orders):
        return orders
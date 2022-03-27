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
        if args.use_cuda:
            self.stale_grad_matrix = self.stale_grad_matrix.cuda()
            self.avg_grad = self.avg_grad.cuda()
        self._reset_random_proj_matrix()
        
    def report_progress(self):
        print(self.timer.summary())
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_greedy
    
    def _reset_random_proj_matrix(self):
        rs = random.randint(0, 10000)
        self.rp = random_projection.SparseRandomProjection(n_components=self.grad_dimen, random_state=rs)
    
    def update_stale_grad(self, optimizer, batch_idx, epoch, add_to_avg=True):
        tensor = flatten_grad(optimizer)
        if self.args.use_random_proj:
            if self.args.use_cuda:
                tensor = tensor.cpu()
            with self.timer("random projection", epoch=epoch):
                tensor = torch.from_numpy(self.rp.fit_transform(tensor.reshape(1, -1)))
            if self.args.use_cuda:
                tensor = tensor.cuda()
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
        X.add_(-1 * self.avg_grad)
        remain_ids = set(range(self.num_batches))
        for i in range(1, self.num_batches+1):
            cur_id = -1
            max_norm = float('inf')
            for cand_id in remain_ids:
                cand_norm = torch.norm(
                    X[cand_id] + cur_sum*(i-1)
                ).item()
                if cand_norm < max_norm:
                    max_norm = cand_norm
                    cur_id = cand_id
            remain_ids.remove(cur_id)
            orders[cur_id] = i
            cur_sum.add_(X[cur_id])
        self.avg_grad.zero_()
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders
        
class FreshGradGreedySort(Sort):
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
        if self.args.use_random_proj:
            self.stale_grad_matrix = torch.zeros(num_batches, self.args.zo_batch_size)
            self.avg_grad = torch.zeros(self.args.zo_batch_size)
        else:
            self.stale_grad_matrix = torch.zeros(num_batches, grad_dimen)
            self.avg_grad = torch.zeros(grad_dimen)
        if args.use_cuda:
            self.stale_grad_matrix = self.stale_grad_matrix.cuda()
            self.avg_grad = self.avg_grad.cuda()
        self._reset_random_proj_matrix()
        
    def report_progress(self):
        print(self.timer.summary())
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_greedy
    
    def _reset_random_proj_matrix(self):
        rs = random.randint(0, 10000)
        self.rp = random_projection.SparseRandomProjection(n_components=self.args.proj_target, random_state=rs)
    
    def _update_fresh_grad(self, optimizer, batch_idx, epoch, add_to_avg=True):
        tensor = flatten_grad(optimizer)
        if self.args.use_random_proj:
            if self.args.use_cuda:
                tensor = tensor.cpu()
            with self.timer("random projection", epoch=epoch):
                tensor = torch.from_numpy(self.rp.fit_transform(tensor.reshape(1, -1)))
            if self.args.use_cuda:
                tensor = tensor.cuda()
            self.stale_grad_matrix[batch_idx].copy_(tensor[0])
            if add_to_avg:
                self.avg_grad.add_(tensor[0] / self.num_batches)
        else:
            self.stale_grad_matrix[batch_idx].copy_(tensor)
            if add_to_avg:
                self.avg_grad.add_(tensor / self.num_batches)
        # make sure the same random matrix is used in one epoch
        if batch_idx == self.num_batches - 1 and self.args.use_random_proj:
            self._reset_random_proj_matrix()

    def sort(self, epoch, model, train_batches, optimizer, oracle_type, orders=None, **kwargs):
        if orders is None:
            orders = {i:0 for i in range(self.num_batches)}
        if self._skip_sort_this_epoch(epoch):
            return orders
        self.avg_grad.zero_()
        for i in orders.keys():
            if oracle_type == 'cv':
                _, batch = train_batches[i]
                loss, _, _ = model(batch)
                optimizer.zero_grad()
                loss.backward()
            else:
                raise NotImplementedError
            self._update_fresh_grad(optimizer=optimizer,
                                    batch_idx=i,
                                    epoch=epoch)
        X = self.stale_grad_matrix.clone()
        X.add_(-1 * self.avg_grad)
        cur_sum = torch.zeros_like(self.avg_grad)
        remain_ids = set(range(self.num_batches))
        for i in range(1, self.num_batches+1):
            cur_id = -1
            max_norm = float('inf')
            for cand_id in remain_ids:
                cand_norm = torch.norm(
                    X[cand_id] + cur_sum
                ).item()
                if cand_norm < max_norm:
                    max_norm = cand_norm
                    cur_id = cand_id
            remain_ids.remove(cur_id)
            orders[cur_id] = i
            cur_sum.add_(X[cur_id])
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders

class ZerothOrderGreedySort(Sort):
    def __init__(self,
                args,
                num_batches,
                grad_dimen,
                model,
                timer=None):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.timer = timer
        self.model = model
        assert self.timer is not None

        self.zo_query_points = [dict() for _ in range(self.args.zo_batch_size)]
        self._init_zo_query_points()
    
    def _init_zo_query_points(self):
        Z = torch.normal(0, 1, size=(self.grad_dimen, self.args.zo_batch_size), 
                            requires_grad=False)
        Q, _ = torch.linalg.qr(Z)
        cur_index = 0
        for name, param in self.model.named_parameters():
            param_size = param.numel()
            for i in range(self.args.zo_batch_size):
                self.zo_query_points[i][name] = Q[:, i][
                    cur_index:cur_index+param_size
                ].clone().reshape(param.shape).to(param.device)
            cur_index += param_size

    def report_progress(self):
        print(self.timer.summary())
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_greedy

    def sort(self, epoch, model, train_batches, oracle_type='cv', orders=None):
        if orders is None:
            orders = {i:0 for i in range(self.num_batches)}
        if self._skip_sort_this_epoch(epoch):
            return orders
        with torch.no_grad():
            mu = 1e-3
            X = [
                torch.zeros(self.args.zo_batch_size)
                for _ in range(len(train_batches))
            ]
            Loss_F_i = [0 for _ in range(len(train_batches))]
            for i in orders.keys():
                # compute all the f_i(x) and store them in the X
                if oracle_type == 'cv':
                    _, batch = train_batches[i]
                    loss, _, _ = model(batch)
                    Loss_F_i[i] = loss.item()
                else:
                    raise NotImplementedError
            for zo_bsz in range(self.args.zo_batch_size):
                for name, param in model.named_parameters():
                    # u  = torch.normal(0, 1, p.data.shape, requires_grad=False, device=p.device)
                    param.data.add_(self.zo_query_points[zo_bsz][name] * mu)
                for i in orders.keys():
                    if oracle_type == 'cv':
                        _, batch = train_batches[i]
                        loss, _, _ = model(batch)
                    else:
                        raise NotImplementedError
                    X[i][zo_bsz] = self.grad_dimen * (loss.item() / mu - Loss_F_i[i] / mu)
                for name, param in model.named_parameters():
                    param.data.add_(-1* self.zo_query_points[zo_bsz][name] * mu)
            avg_query = sum(X) / len(X)
            cur_sum = torch.zeros_like(avg_query)
            X.add_(-1 * avg_query)
            remain_ids = set(range(self.num_batches))
            for i in range(self.num_batches):
                cur_id = -1
                max_norm = float('inf')
                for cand_id in remain_ids:
                    cand_norm = torch.norm(
                        X[cand_id] + cur_sum
                    ).item()
                    if cand_norm < max_norm:
                        max_norm = cand_norm
                        cur_id = cand_id
                remain_ids.remove(cur_id)
                orders[cur_id] = i
                cur_sum.add_(X[cur_id])
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders

class FreshGradDiscrepencyMinimizationSort(Sort):
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
        if self.args.use_random_proj:
            self.stale_grad_matrix = torch.zeros(num_batches, self.args.zo_batch_size)
            self.avg_grad = torch.zeros(self.args.zo_batch_size)
        else:
            self.stale_grad_matrix = torch.zeros(num_batches, grad_dimen)
            self.avg_grad = torch.zeros(grad_dimen)
        if args.use_cuda:
            self.stale_grad_matrix = self.stale_grad_matrix.cuda()
            self.avg_grad = self.avg_grad.cuda()
        self._reset_random_proj_matrix()
        
    def report_progress(self):
        print(self.timer.summary())
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_greedy
    
    def _reset_random_proj_matrix(self):
        rs = random.randint(0, 10000)
        self.rp = random_projection.SparseRandomProjection(n_components=self.args.proj_target, random_state=rs)
    
    def _update_fresh_grad(self, optimizer, batch_idx, epoch, add_to_avg=True):
        tensor = flatten_grad(optimizer)
        if self.args.use_random_proj:
            if self.args.use_cuda:
                tensor = tensor.cpu()
            with self.timer("random projection", epoch=epoch):
                tensor = torch.from_numpy(self.rp.fit_transform(tensor.reshape(1, -1)))
            if self.args.use_cuda:
                tensor = tensor.cuda()
            self.stale_grad_matrix[batch_idx].copy_(tensor[0])
            if add_to_avg:
                self.avg_grad.add_(tensor[0] / self.num_batches)
        else:
            self.stale_grad_matrix[batch_idx].copy_(tensor)
            if add_to_avg:
                self.avg_grad.add_(tensor / self.num_batches)
        # make sure the same random matrix is used in one epoch
        if batch_idx == self.num_batches - 1 and self.args.use_random_proj:
            self._reset_random_proj_matrix()

    def sort(self, epoch, model, train_batches, optimizer, oracle_type, orders=None, **kwargs):
        if orders is None:
            orders = {i:0 for i in range(self.num_batches)}
        if self._skip_sort_this_epoch(epoch):
            return orders
        self.avg_grad.zero_()
        for i in orders.keys():
            if oracle_type == 'cv':
                _, batch = train_batches[i]
                loss, _, _ = model(batch)
                optimizer.zero_grad()
                loss.backward()
            else:
                raise NotImplementedError
            self._update_fresh_grad(optimizer=optimizer,
                                    batch_idx=i,
                                    epoch=epoch)
        X = self.stale_grad_matrix.clone()
        X.add_(-1 * self.avg_grad)
        cur_sum = torch.zeros_like(self.avg_grad)
        cur_id = -1
        max_norm = float('inf')
        for cand_id in set(range(self.num_batches)):
            cand_norm = torch.norm(
                X[cand_id]
            ).item()
            if cand_norm < max_norm:
                max_norm = cand_norm
                cur_id = cand_id
        X[cur_id], X[0] = X[0], X[cur_id]
        signs = [] # output
        w = torch.zeros_like(X[0])
        for x in X:
            prob = 0.5 - torch.dot(w, x)
            prob = 0 if prob < 0 else prob
            prob = 1 if prob > 1 else prob
            if prob < 0.5:
                sign = -1
            else:
                sign = 1
            signs.append(sign)
            w = w + sign*x

        last = self.num_batches
        first = 0
        for i in range(self.num_batches):
            if signs[i] > 0:
                orders[i] = first
                first += 1
            else:
                orders[i] = last
                last -= 1
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders


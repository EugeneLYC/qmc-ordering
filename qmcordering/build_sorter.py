from .constants import _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_
from .sort.algo import StaleGradGreedySort, ZerothOrderGreedySort

def get_sorter(args,
            loader,
            dimension,
            timer):
    if args.shuffle_type == _STALE_GRAD_SORT_:
        num_batches = len(list(enumerate(loader)))
        sorter = StaleGradGreedySort(args,
                                    num_batches,
                                    grad_dimen=dimension,
                                    timer=timer)
    elif args.shuffle_type == _ZEROTH_ORDER_SORT_:
        num_batches = len(list(enumerate(loader)))
        sorter = ZerothOrderGreedySort(args,
                                    num_batches,
                                    grad_dimen=dimension,
                                    timer=timer)
    else:
        raise NotImplementedError("This method does not need sorter or is not supported yet")
    return sorter
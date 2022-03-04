from .sort.algo import StaleGradGreedySort

def get_sorter(args,
            loader,
            dimension,
            timer):
    if args.shuffle_type == 'greedy':
        num_batches = len(list(enumerate(loader)))
        sorter = StaleGradGreedySort(args,
                                    num_batches,
                                    grad_dimen=dimension,
                                    timer=timer)
    else:
        raise NotImplementedError("This method does not need sorter or is not supported yet")
    return sorter
from multiprocessing import Pool


def multi_process_lst(lst, apply_on_chunk, chunk_size=1000, n_processes=1, args=None):
    '''
    applies apply_on_chunk on lst using n_processes each gets chunk_size items from lst each time
    '''
    chunks = split(lst, n_processes)
    chunks = flatten_iterable(group(c, chunk_size) for c in chunks if len(c) > 0)
    if n_processes == 1:
        for c in lst:
            yield apply_on_chunk([c], *args)
    else:
        with Pool(n_processes) as pool:

            for preproc_inst in pool.imap_unordered(unpack_args_wrapper,
                                                    [(apply_on_chunk, (c, *args)) for c in chunks]):
                yield preproc_inst


def unpack_args_wrapper(function_args_tup):
    return function_args_tup[0](*function_args_tup[1])


def flatten_iterable(listoflists):
    return [item for sublist in listoflists for item in sublist]


def split(lst, n_groups):
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst, max_group_size):
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

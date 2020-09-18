import numpy as np


def negative_sampling(pos_ids, num_items, sample_size=10):
    """negative sample for candidate generation. assumes pos_ids is ordered."""
    raw_sample = np.random.randint(0, num_items - len(pos_ids), size=sample_size)
    pos_ids_adjusted = pos_ids - np.arange(0, len(pos_ids))
    ss = np.searchsorted(pos_ids_adjusted, raw_sample, side='right')
    neg_ids = raw_sample + ss
    return neg_ids

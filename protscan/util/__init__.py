"""Utility functions."""

import random
from itertools import tee, izip
import numpy as np
from eden.util import selection_iterator, iterator_size

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


def random_partition(int_range, n_splits, random_state=1234):
    """Partition a range in a random way."""
    random.seed(random_state)
    ids = range(int_range)
    random.shuffle(ids)
    split_points = \
        [int(int_range * (float(i) / n_splits)) for i in range(1, n_splits)]
    return np.split(ids, split_points)


def random_partition_iter(iterable, n_splits, random_state=1234):
    """Partition a generator in a random way (should mantain the unbalance)."""
    size_iterable, iterable = tee(iterable)
    size = iterator_size(iterable)
    part_ids = random_partition(size, n_splits=n_splits,
                                random_state=random_state)
    parts = list()
    for p in part_ids:
        iterable, iterable_ = tee(iterable)
        parts.append(selection_iterator(iterable_, p))
    return parts


def balanced_split(sequences, bin_sites, n_splits,
                   random_state=1234):
    """Balanced split over binding/non-binding sequences."""
    # find the transcript names of positive and negatives
    sequences, sequences_ = tee(sequences)
    pos_ids = list()
    neg_ids = list()
    for i, (attr, _) in enumerate(sequences_):
        tr_name = attr['tr_name']
        is_binding = bin_sites.get(tr_name, False)
        if is_binding:
            pos_ids.append(i)
        else:
            neg_ids.append(i)

    random.seed(random_state)
    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    pos_split_points = \
        [int(len(pos_ids) * (float(i) / n_splits)) for i in range(1, n_splits)]
    neg_split_points = \
        [int(len(neg_ids) * (float(i) / n_splits)) for i in range(1, n_splits)]

    parts = list()
    for pos, neg in izip(np.split(pos_ids, pos_split_points),
                         np.split(neg_ids, neg_split_points)):
        sequences, sequences_ = tee(sequences)
        parts.append(selection_iterator(
            sequences_, np.concatenate([pos, neg])))
    return parts


def balanced_fraction(sequences, bin_sites, opt_fraction=1.0,
                      random_state=1234):
    """Balanced sample of sequences (over binding/non-binding)."""
    # find the transcript names of positive and negatives
    sequences, sequences_ = tee(sequences)
    pos_names = list()
    neg_names = list()
    for attr, _ in sequences_:
        tr_name = attr['tr_name']
        is_binding = bin_sites.get(tr_name, False)
        if is_binding:
            pos_names.append(tr_name)
        else:
            neg_names.append(tr_name)
    # sample from positives and negatives
    selected = list()
    random.seed(random_state)
    k_pos = max(1, int(opt_fraction * len(pos_names)))
    selected.extend(random.sample(pos_names, k_pos))
    k_neg = max(1, int(opt_fraction * len(neg_names)))
    selected.extend(random.sample(neg_names, k_neg))
    # yield only sequences in selected
    for attr, s in sequences:
        tr_name = attr['tr_name']
        if tr_name in selected:
            yield attr, s


def add_default(default_value, l):
    """Add default value as first value of the list."""
    return np.insert(l, 0, default_value)


def random_exp_scale(from_exp, to_exp, size=1):
    """Random values in exponential scale."""
    sample = list()
    for _ in range(size):
        s = np.random.uniform(0.1, 1.0) * \
            10 ** np.random.randint(from_exp + 1, to_exp)
        sample.append(s)
    return np.array(sample)


def additive_update(dic, to_add):
    """Update a dictionary (summing values if the key exists)."""
    for k, v in to_add.iteritems():
        if k in dic.keys():
            dic[k] += v
        else:
            # the values are numpy arrays
            dic.update({k: v.copy()})

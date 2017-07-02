"""Peak extraction."""

import random
import numpy as np
from eden.util import compute_intervals
from eden import apply_async
from protscan.util.mean_shift import mean_shift
import multiprocessing as mp
from scipy import optimize

import logging
logger = logging.getLogger(__name__)

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016-2017, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


def split_keys(profiles, bin_sites, random_state=1234):
    """Balanced split over binding/non-binding sequences."""
    random.seed(random_state)
    pos_keys = bin_sites.keys()
    neg_keys = list(set(profiles.keys()) - set(pos_keys))
    random.shuffle(pos_keys)
    random.shuffle(neg_keys)

    len_pos = len(pos_keys)
    pos_keys1 = pos_keys[:len_pos / 2]
    pos_keys2 = pos_keys[len_pos / 2:]

    len_neg = len(neg_keys)
    neg_keys1 = neg_keys[:len_neg / 2]
    neg_keys2 = neg_keys[len_neg / 2:]

    return [pos_keys1, pos_keys2, neg_keys1, neg_keys2]


def _find_blocks(signal, window):
    """Combine local maxima and minima to define blocks in a signal."""
    minima = mean_shift(signal, window, mode='min')
    maxima = mean_shift(signal, window, mode='max')

    blocks = set()
    for i in range(minima.shape[0] - 1):
        if np.any(np.logical_and(maxima > minima[i], maxima < minima[i + 1])):
            start = minima[i]
            end = minima[i + 1]
            blocks.add((start, end, max(signal[start:end])))
    return np.array(sorted(list(blocks)))


def serial_find_blocks(profiles, window):
    """Find blocks in profiles (serial version)."""
    blocks = {k: _find_blocks(profiles[k], window) for k in profiles}
    return blocks


def multiprocess_find_blocks(profiles, window, n_blocks=None,
                             block_size=None, n_jobs=-1):
    """Find blocks in profiles (parallel version)."""
    intervals = compute_intervals(
        size=len(profiles), n_blocks=n_blocks, block_size=block_size)
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)
    results = [apply_async(pool, serial_find_blocks,
                           args=(dict(profiles.items()[start:end]),
                                 window))
               for start, end in intervals]
    dicts = [p.get() for p in results]
    pool.close()
    pool.join()
    blocks = {k: v for d in dicts for k, v in d.items()}
    return blocks


def find_blocks(profiles, window, n_blocks=None, block_size=None,
                n_jobs=-1):
    """Find blocks for each transcript in profiles."""
    if n_jobs == 1:
        blocks = serial_find_blocks(profiles, window)
    else:
        blocks = multiprocess_find_blocks(
            profiles, window, block_size=20, n_jobs=n_jobs)

    return blocks


def func(x, a, b):
    """Function used to fit the cumulative distribution of the peaks."""
    return 1. / (1. + np.exp((-x - a) / b))


def fit_optimal(blocks, keys):
    """Fit the cumulative distribution, by optimization."""
    values = list()
    for k in keys:
        values.extend([max_val for (_, _, max_val) in blocks[k]])
    hist, bin_edges = np.histogram(values, bins=100, normed=True)
    bin_centers = np.array(
        [(bin_edges[i - 1] + bin_edges[i]) / 2
         for i in range(1, len(bin_edges))])
    cumulative = np.cumsum(hist) / np.sum(hist)
    popt, _ = optimize.curve_fit(func, bin_centers, cumulative)
    return popt


def select_blocks(blocks, fit_keys, pre_keys, pval_threshold):
    """Select blocks with pval less or equal than the threshold."""
    selected = list()
    popt = fit_optimal(blocks, fit_keys)
    for transcript in pre_keys:
        for (start, end, high_val) in blocks[transcript]:
            pval = 1 - func(high_val, popt[0], popt[1])
            if pval <= pval_threshold:
                selected.append((transcript, int(start), int(end), pval))
    return selected


def find_peaks(profiles, bin_sites, pval_threshold=0.1, window=10,
               sorted_by_pval=False, random_state=1234, n_jobs=-1):
    """Find significant peaks in the profiles."""
    # indeces
    pos1 = 0
    pos2 = 1
    neg1 = 2
    neg2 = 3

    keys = split_keys(profiles, bin_sites, random_state)
    blocks = find_blocks(profiles, window, block_size=20, n_jobs=n_jobs)

    selected = list()
    selected.extend(select_blocks(blocks, keys[neg1], keys[
                    pos2] + keys[neg2], pval_threshold))
    selected.extend(select_blocks(blocks, keys[neg2], keys[
                    pos1] + keys[neg1], pval_threshold))

    if not sorted_by_pval:
        # sort by transcript name and position
        selected = sorted(selected)
    else:
        selected = sorted(selected, key=lambda x: (x[3], x[0], x[1], x[2]))

    return selected

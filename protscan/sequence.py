"""Functions for sequence based target prediction."""

from itertools import izip, tee
import random
from math import ceil
import numpy as np

import protscan.common as common

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


def get_positive_subseq(bin_center, max_dist, split_window, tr_len):
    """Compute the start and end coordinates of a positive subsequence.

    Parameters
    ----------
    bin_center : int
        Cernter of a binding site.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    split_window : int
        Window size of the split_iterator.

    tr_len : int
        Lenght of the trascript.

    Returns
    -------
    start : int
        Starting position of the positive subsequence.
    end : int
        Ending position of the positive subsequence.
    """
    start = max(0, bin_center - max_dist - split_window / 2 + 1)
    end = min(bin_center + max_dist +
              int(ceil(float(split_window) / 2)) - 1, tr_len)
    return start, end


def get_left_border_subseq(prevbin_center, bin_center, max_dist, split_window):
    """Compute the start and end coordinates of a left border subsequence.

    Parameters
    ----------
    prevbin_center : int
        Center of the previous binding site.

    bin_center : int
        Center of the binding site.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    split_window : int
        Window size of the split_iterator.

    Returns
    -------
    start : int
        Starting position of the left border subsequence (None if there is
        no space for the border split, e.g. another binding site is too close).
    end : int
        Ending position of the left border subsequence. (None if there is
        no space for the border split, e.g. another binding site is too close).
    """
    # ASSUMPTION: bins are sorted, no binding sites in between
    if prevbin_center < bin_center:
        if prevbin_center is None:
            left_limit = 0
        else:
            left_limit = prevbin_center + max_dist + \
                int(ceil(float(split_window) / 2))
        end = bin_center - max_dist + int(ceil(float(split_window) / 2))
        start = end - max_dist
        if start >= left_limit:
            return start, end
    return None, None


def get_right_border_subseq(bin_center, nextbin_center, max_dist, split_window,
                            tr_len):
    """Compute the start and end coordinates of a left border subsequence.

    Parameters
    ----------
    bin_center : int
        Center of the binding site.

    nextbin_start : int
        Center of the next binding site.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    split_window : int
        Window size of the split_iterator.

    tr_len : int
        Lenght of the trascript.

    Returns
    -------
    start : int
        Starting position of the right border subsequence (None if there is
        no space for the border split, e.g. another binding site is too close).
    end : int
        Ending position of the right border subsequence. (None if there is
        no space for the border split, e.g. another binding site is too close).
    """
    # ASSUMPTION: bins are sorted, no binding sites in between
    if bin_center < nextbin_center or nextbin_center is None:
        if nextbin_center is None:
            right_limit = tr_len
        else:
            right_limit = nextbin_center - max_dist - split_window / 2
        start = bin_center + max_dist - split_window / 2
        end = start + max_dist
        if end <= right_limit:
            return start, end
    return None, None


def get_negative_subseqs(bin1_center, bin2_center, max_dist, split_window,
                         tr_len, negative_ratio=1, random_state=1234):
    """Negative subsequences between binding sites.

    Parameters
    ----------
    bin1_center : int
        Center of a binding site.

    bin2_center : int
        Center of the consecutive binding site of bin1. I.e. no
        binding sites in between.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    split_window : int
        Window size of the split_iterator.

    tr_len : int
        Lenght of the trascript.

    negative_ratio : float (default : 1)
        Maximum nucleotide ratio for the negative splits. This parameter
        controls the number of the negative splits to be sampled.
        The total length (in nucleotides) of all the negative splits
        is computed as: max_dist * negative_ratio
        Negative splits may overlap.

    random_state : int (default : 1234)
        Seed for RNG that controls the sampling of the negative splits.

    Returns
    -------
    splits : list
        List of (start, end) potions of the negative subsequences.
    """
    # ASSUMPTION: bins are sorted, no binding sites in between
    if bin1_center < bin2_center or bin2_center is None:
        if bin1_center is None:
            left_limit = 0
        else:
            left_limit = bin1_center + 2 * max_dist - split_window / 2
        if bin2_center is None:
            right_limit = tr_len
        else:
            right_limit = bin2_center - 2 * max_dist + \
                int(ceil(float(split_window) / 2))
        neg_window = max(0, right_limit - left_limit)
        if neg_window >= 2 * max_dist + split_window - 2:
            choices = list()
            for _ in range(negative_ratio):
                choices.append(
                    int(random.triangular(left_limit + max_dist +
                                          split_window / 2,
                                          right_limit - max_dist -
                                          split_window / 2)))

            splits = [(c - max_dist - split_window / 2 + 1, c +
                       max_dist + split_window / 2 - 1) for c in choices]
            return splits
    return [(None, None)]


def train_selector(sequences, bin_sites, max_dist, random_state=1234,
                   **params):
    """Select training subsequences from RNA sequences.

    Parameters
    ----------
    sequences : iterable
        RNA sequences (yielded from fasta_to_seq).

    bin_sites : dict
        Binding site regions (from bed_to_dictionary).

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    random_state : int (default : 1234)
        Seed for RNG that controls the sampling of the negative splits.

    **params : dict
        Dictionary of pre_processing parameters.

    Returns
    -------
    attr : dict
        Sequence information. tr_name: name of the sequence in the fasta file,
        tr_length: lenght of the transcript, lenght: lenght of the subsequence,
        start: starting position of the subsequence, end: ending position of
        the subsequence, type: class of the subsequence (either 'POS', 'NEG' or
        'BOR').
    subseq : str
        Subsequence.
    """
    negative_ratio = params.get('negative_ratio', 1)
    split_window = params.get('split_window', 50)

    non_binding = 0
    negative_remainder = 0
    sequences, sequences_ = tee(sequences)

    for attr, seq in sequences_:
        bins = bin_sites.get(attr['tr_name'].split('.')[0], False)
        if bins is False:
            non_binding += 1
            continue
        tr_len = attr['tr_len']
        # iterate over couples of consecutive binding sites (bins are sorted)
        iterator = izip([(None, None, None)] + bins[:-1], bins, bins[1:] +
                        [(None, None, None)])
        for (_, _, prevbin_center), (_, _, bin_center),\
                (_, _, nextbin_center) in iterator:
            for start, end in get_negative_subseqs(
                    prevbin_center, bin_center, max_dist, split_window,
                    tr_len, negative_ratio, random_state):
                if start is not None and end is not None:
                    subseq = seq[start:end]
                    new_attr = attr.copy()
                    new_attr['start'] = start
                    new_attr['end'] = end
                    new_attr['length'] = end - start
                    new_attr['type'] = 'NEG'
                    yield new_attr, subseq
                else:
                    negative_remainder += negative_ratio

            # left border
            start, end = get_left_border_subseq(
                prevbin_center, bin_center, max_dist, split_window)
            if start is not None and end is not None:
                subseq = seq[start:end]
                new_attr = attr.copy()
                new_attr['start'] = start
                new_attr['end'] = end
                new_attr['length'] = end - start
                new_attr['type'] = 'BOR'
                yield new_attr, subseq

            # positive
            start, end = get_positive_subseq(
                bin_center, max_dist, split_window, tr_len)
            subseq = seq[start:end]
            new_attr = attr.copy()
            new_attr['start'] = start
            new_attr['end'] = end
            new_attr['length'] = end - start
            new_attr['type'] = 'POS'
            yield new_attr, subseq

            # right border
            start, end = get_right_border_subseq(
                bin_center, nextbin_center, max_dist, split_window, tr_len)
            if start is not None and end is not None:
                subseq = seq[start:end]
                new_attr = attr.copy()
                new_attr['start'] = start
                new_attr['end'] = end
                new_attr['length'] = end - start
                new_attr['type'] = 'BOR'
                yield new_attr, subseq

            # # negative (only for the last iteration)
            # if nextbin_center is None:
            #     for start, end in get_negative_subseqs(
            #             bin_center, nextbin_center, max_dist, split_window,
            #             tr_len, negative_ratio, random_state):
            #         if start is not None and end is not None:
            #             subseq = seq[start:end]
            #             new_attr = attr.copy()
            #             new_attr['start'] = start
            #             new_attr['end'] = end
            #             new_attr['length'] = end - start
            #             new_attr['type'] = 'NEG'
            #             yield new_attr, subseq
            #         else:
            #             negative_remainder += negative_ratio

    # account for the missing negative sequences
    if negative_remainder > 0 and non_binding > 0:
        new_negative_ratio = int(
            ceil(float(negative_remainder) / non_binding))
        for attr, seq in sequences:
            bins = bin_sites.get(attr['tr_name'].split('.')[0], False)
            tr_len = attr['tr_len']
            if bins is False and negative_remainder > 1:
                for start, end in get_negative_subseqs(
                        None, None, max_dist, split_window, tr_len,
                        new_negative_ratio, random_state):
                    if start is not None and end is not None:
                        subseq = seq[start:end]
                        new_attr = attr.copy()
                        new_attr['start'] = start
                        new_attr['end'] = end
                        new_attr['length'] = end - start
                        new_attr['type'] = 'NEG'
                        negative_remainder -= 1
                        yield new_attr, subseq


def split_iterator(iterable, **params):
    """Split sequences.

    Parameters
    ----------
    iterable : iterable
        RNA sequences to split.

    **params : dict
        Dictionary of pre_processing parameters.

    Returns
    -------
    attr_out : dict
        Sequence information. tr_name: name of the sequence in the fasta file,
        tr_length: lenght of the transcript, lenght: lenght of the subsequence,
        start: starting position of the subsequence, end: ending position of
        the subsequence, center: center position of the subsequence
        type: class of the subsequence (either 'POS', 'NEG' or
        'BOR').

    seq_out : str
        Subsequence.
    """
    step = params.get('split_step', 1)
    window = params.get('split_window', 50)
    for attr, seq in iterable:
        seq_len = len(seq)
        if seq_len >= window:
            for start in range(0, seq_len, step):
                seq_out = seq[start: start + window]
                if len(seq_out) == window:
                    attr_out = attr.copy()
                    if 'start' in attr_out.keys():
                        attr_out['start'] += start
                    else:
                        attr_out['start'] = start
                    attr_out['end'] = attr_out['start'] + window
                    attr_out['length'] = window
                    attr_out['center'] = common.center(
                        attr_out['start'], attr_out['end'])
                    yield (attr_out, seq_out)


def _min_dist(center, bins):
    """Compute the minumum distance from a binding site.

    Parameters
    ----------
    center : int
        Center index of a RNA sequence.
    bins : list
        List of binding sites of the RNA sequence.

    Returns
    -------
    best : int
        Distance of center from the closest binding site in bins.

    Raises
    ------
    IndexError
        Raises if bins is empty, i.e. the RNAsequence has no binding sites.

    >>> _min_dist(45, [(20,30,25), (500, 530, 515)])
    20
    """
    # ASSUMPTION: bins are sorted, no binding sites in between
    _, _, c = bins[0]
    best = abs(center - c)
    for _, _, c in bins[1:]:
        dist = abs(center - c)
        if dist < best:
            best = dist
        else:
            break
    return best


def add_distance(subsequences, bin_sites):
    """Per subsequence distance from the closest binding site.

    NOTE: sequence based add_distance is suppoded to be called after the
    split_iterator (on the RNA subsequences).

    Parameters
    ----------
    subsequences : iterable
        RNA subsequences (processed by the split_iterator).

    bin_sites : dict
        Binding site regions (from bed_to_dictionary).

    Returns
    -------
    attr : dict
        Sequence information. Added 'dist'.

    subseq: str
        Subsequence.
    """
    for attr, subseq in subsequences:
        try:
            bins = bin_sites[attr['tr_name'].split('.')[0]]
            seq_center = attr['center']
            dist = _min_dist(seq_center, bins)
            attr['dist'] = dist
        except Exception:
            attr['dist'] = None
        yield attr, subseq


def sequence_preprocessor(iterable, which_set, bin_sites=None, max_dist=None,
                          random_state=1234, **params):
    """Preprocess sequences."""
    assert which_set == 'train' or which_set == 'test', \
        "which_set must be either 'train' or 'test'."

    if which_set == 'train':
        iterable = train_selector(
            iterable, bin_sites, max_dist, random_state, **params)

    split = split_iterator(iterable, **params)

    if which_set == 'train':
        split = add_distance(split, bin_sites)

    return split


def print_stats(iterable):
    """Print number of positive, border, and negative subsequences."""
    pos = bor = neg = 0
    for attr, seq in iterable:
        if attr['type'] == 'POS':
            pos += 1
        elif attr['type'] == 'BOR':
            bor += 1
        elif attr['type'] == 'NEG':
            neg += 1
        else:
            raise Exception("ERROR: unrecognized subsequence type:" +
                            str(attr['type']))
    print "POS: %i\nBOR: %i\nNEG: %i\nTOT: %i" %\
        (pos, bor, neg, pos + bor + neg)


def vote_aggregator(pred_vals, info, max_dist, **params):
    """Get vote profiles from predicted distances and subseq info.

    The function assumes that the input contains complete data (predictions
    from all subsequences) for one or more ENTIRE transcripts.

    Params
    ------
    pred : np.array (dtype : float)
        Predicted regression values, from SGDRegressor.

    info : dict
        Dictionary containing subsequence information for each predicted
        distance in pred. The dictionary has 3 keys 'tr_name', 'tr_len' and
        'center' which is the center of the subsequence in the full
        transcript.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    **params : dict
        Dictionary of prediction parameters.

    Returns
    -------
    votes : dict
        Dictionary of predicted votes. The key is the transcript
        name, the value is an np.array containing one float value per
        nucleotide
    """
    pred_dists = [common.val_to_dist(v, max_dist) for v in pred_vals]
    votes = dict()
    for pv, pd, i in izip(pred_vals, pred_dists, info):
        tr_name = i['tr_name']
        tr_len = i['tr_len']
        center = i['center']
        if tr_name not in votes:
            votes[tr_name] = np.zeros(tr_len)
        if pd < max_dist:
            downstream = center - pd
            if downstream >= 0:
                votes[tr_name][downstream] += pv
            upstream = center + pd
            if upstream < tr_len:
                votes[tr_name][downstream] += pv
    return votes

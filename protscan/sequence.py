"""Functions for sequence based target prediction."""

from itertools import izip
import numpy as np

import protscan.common as common

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


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


def add_supervision(subsequences, bin_sites, max_dist):
    """Per subsequence distance from the closest binding site.

    NOTE: sequence based add_distance is suppoded to be called after the
    split_iterator (on the RNA subsequences).

    Parameters
    ----------
    subsequences : iterable
        RNA subsequences (processed by the split_iterator).

    bin_sites : dict
        Binding site regions (from bed_to_dictionary).

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    Returns
    -------
    attr : dict
        Sequence information. Added 'dist' and 'type'.

    subseq: str
        Subsequence.
    """
    for attr, subseq in subsequences:
        tr_name = attr['tr_name'].split('.')[0]
        bins = bin_sites.get(tr_name, False)
        if bins is not False:
            seq_center = attr['center']
            dist = _min_dist(seq_center, bins)
            attr['dist'] = dist
            if dist < max_dist:
                attr['type'] = 'POS'
            elif dist == max_dist:
                attr['type'] = 'BOR'
            else:
                attr['type'] = 'NEG'
        else:
            attr['dist'] = None
            attr['type'] = 'NEG'
        yield attr, subseq


def sequence_preprocessor(iterable, which_set, bin_sites=None, max_dist=None,
                          random_state=1234, **params):
    """Preprocess sequences."""
    assert which_set == 'train' or which_set == 'test', \
        "which_set must be either 'train' or 'test'."

    split = split_iterator(iterable, **params)

    if which_set == 'train':
        split = add_supervision(split, bin_sites, max_dist)

    return split


def get_stats(iterable):
    """Get number of positive, border, and negative subsequences."""
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
    return pos, bor, neg


def vote_aggregator(pred_vals, info, max_dist):
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

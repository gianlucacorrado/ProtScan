"""Common functions."""

from scipy.signal import gaussian
from numpy import convolve
from eden.modifier.fasta import fasta_to_fasta

import logging
logger = logging.getLogger(__name__)

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


def fasta_to_seq(input):
    """Load sequences tuples from fasta file.

    Parameters
    ----------
    input : str
        Fasta file.

    Returns
    -------
    attr : dict
        Sequence information. tr_name: name of the sequence in the fasta file,
        tr_length: lenght of the transcript, lenght: lenght of the sequence.
        After fasta_to_seq tr_len == lenght.
    seq : str
        Sequence.
    """
    lines = fasta_to_fasta(input)
    for line in lines:
        attr = dict()
        attr['tr_name'] = line.split('.')[0]
        seq = lines.next()
        attr['tr_len'] = len(seq)
        attr['length'] = len(seq)
        if len(seq) == 0:
            raise Exception("ERROR: empty sequence")
        yield attr, seq


def center(start, end):
    """Compute the position of the center of a subsequence.

    Parameters
    ----------
    start : int
        Starting position.

    end : int
        Ending position

    Returns
    -------
    center : int
        Central point between start and end.

    >>> center(4, 17)
    12

    >>> center(4, 14)
    9
    """
    center = start + (end - start) / 2
    return center


def bed_to_dictionary(bed, less_equal=None, greater_equal=None):
    """Build a dictionary with the start, stop of the binding sites.

    Parameters
    ----------
    bed : str
        Bed file.

    less_equal : float (default : None)
        Select rows in the BED file with score value (5th column) less or
        equal to the specified value. None means no filtering.

    greater_equal : float (default : None)
        Select rows in the BED file with score value (5th column) greater or
        equal to the specified value. None means no filtering.

    Returns
    -------
    bin_sites : dict
        Dictionary of triplets (bin_start, bin_end, center). One entry per
        transcript.
    """
    bin_sites = dict()
    n_bin_sites = n_kept = 0
    f = open(bed)
    for line in f:
        n_bin_sites += 1
        if greater_equal is not None or less_equal is not None:
            try:
                # if the 5th column is a score
                score = float(line.strip().split()[4])
            except:
                continue
            else:
                # skip line if score does not satisfy the conditions
                if less_equal is not None:
                    if score > less_equal:
                        continue

                if greater_equal is not None:
                    if score < greater_equal:
                        continue

        n_kept += 1
        seq_name = line.strip().split()[0]
        bin_start = int(line.strip().split()[1])
        bin_end = int(line.strip().split()[2])
        if seq_name in bin_sites.keys():
            bin_sites[seq_name].append((bin_start, bin_end,
                                        center(bin_start, bin_end)))
        else:
            bin_sites[seq_name] = [(bin_start, bin_end,
                                    center(bin_start, bin_end))]
    f.close()
    # sort the binding sites
    for k, v in bin_sites.iteritems():
        bin_sites[k] = sorted(v)

    logger.debug("Imported %d of %d binding sites." % (n_kept, n_bin_sites))
    return bin_sites


def dist_to_val(d, max_dist):
    """From distance to regression value."""
    if d is None or d >= max_dist:
        return 0.
    else:
        return 1. - (float(d) / max_dist)


def val_to_dist(v, max_dist):
    """From regression value to distance."""
    return int(max_dist - v * max_dist)


def smooth(votes, **params):
    """Compute the convolution with a Gaussian signal."""
    window = params.get('window', 50)
    std = params.get('std', 20)
    profiles = dict()
    window = gaussian(window, std=std)
    for k, vote in votes.iteritems():
        smoothed = convolve(vote, window, mode='same')
        profiles[k] = smoothed
    return profiles

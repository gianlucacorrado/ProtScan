"""Mean shift based peak extractor."""

import numpy as np

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016-2017, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"

def boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of data.

    Modified from the original code of scipy. To detect flat maxima or minima
    regions.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take 2 numbers as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.  'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'.  See numpy.take

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as data that is True at an extrema,
        False otherwise.

    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in xrange(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results


def mean_shift(signal, window, mode):
    """Mean shift to detect maxima and minima."""
    assert mode == 'min' or mode == 'max', \
        "Invalid mode %s, select either 'min' or 'max'" % mode
    points = set(np.arange(signal.shape[0]))
    new_points = set()
    while True:
        for p in points:
            start = max(0, p - window)
            end = min(p + window, signal.shape[0] - 1)
            subsig = signal[start:end]
            if mode == 'min':
                toadd = [i for i in np.where(
                    ~boolrelextrema(subsig, np.greater_equal, order=1))[0]
                    if subsig[i] == subsig.min()]
            else:
                toadd = [i for i in np.where(
                    ~boolrelextrema(subsig, np.less_equal, order=1))[0]
                    if subsig[i] == subsig.max()]
            for a in toadd:
                new_points.add(a + start)

        if new_points == points:
            return np.array(sorted(list(points)))
        else:
            points = new_points
            new_points = set()

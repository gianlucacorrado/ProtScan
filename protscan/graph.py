"""Functions for graph based target prediction."""

from itertools import izip, tee, groupby
from operator import itemgetter
from math import ceil
import random
import numpy as np
from collections import deque

from eden.converter.rna.rnafold import rnafold_to_eden
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
# from eden.util import mp_pre_process

import protscan.common as common
from protscan.util.data import HDFDataManager

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


def rnafold(sequences):
    """Fold sequences using RNAfold."""
    graphs = rnafold_to_eden(sequences)
    return graphs


def rnaplfold(sequences):
    """Fold sequences using RNAplfold."""
    params = dict(window_size=250,
                  max_bp_span=150,
                  avg_bp_prob_cutoff=0.01,
                  max_num_edges=2,
                  no_lonely_bps=True,
                  nesting=True,
                  hard_threshold=0.5)
    graphs = rnaplfold_to_eden(sequences, **params)
    return graphs


def _bfs_min_dist(graph, bin_center, max_depth=None):
    """BFS assigning distance from the closest binding site.

    Parameters
    ----------
    graph : NetworkX Graph
        Graph computed with one of the EDeN wrappers for the Vienna RNA
        package.

    bin_center : int
        Node index of the binding site center.

    max_depth : int (default : None)
        Max depth of the BFS. Nodes more distant than max_depth from a
        binding site will have 'dist' set to None. If max_depth is None, then
        the BFS will stop after discovery of all the nodes in the graph.

    Returns
    -------
    graph : NetworkX Graph
        Input graph with added node attribute 'dist', containing the distance
        from the binding center.
    """
    visited = set()
    # q is the queue containing the frontier to be expanded in the BFS
    q = deque()
    q.append(bin_center)
    # the map associates to each vertex id the distance from the root
    dist = {}
    dist[bin_center] = 0
    visited.add(bin_center)
    # add vertex at distance 0
    graph.node[bin_center]['dist'] = 0
    while len(q) > 0:
        # extract the current vertex
        u = q.popleft()
        d = dist[u] + 1
        if max_depth is None or d <= max_depth:
            # iterate over the neighbors of the current vertex
            for v in graph.neighbors(u):
                if v not in visited:
                    if graph.edge[u][v].get('nesting', False) is False:
                        dist[v] = d
                        visited.add(v)
                        q.append(v)
                        if graph.node[v]['dist'] is None:
                            graph.node[v]['dist'] = d
                        else:
                            graph.node[v]['dist'] = min(graph.node[v]['dist'],
                                                        d)
    return graph


def add_distance(graphs, bin_sites, max_distance=None):
    """Per node distance from the closest binding site.

    NOTE: graph based add_distance is suppoded to be called before the
    train_selector (on the entire RNA graph).

    Parameters
    ----------
    graphs : iterable
        Graphs computed with one of the EDeN wrappers for the Vienna RNA
        package.

    bin_sites : dict
        Binding site regions (from bed_to_dictionary).

    max_distance : int (default : None)
        Maximum distance from a binding site to add. Distances greater than
        max_distance will be represented with None.

    Returns
    -------
    graph : NetworkX Graph
        Graph with added node attribute 'dist', containing the distance
        from the binding center.
    """
    for graph in graphs:
        # initialize distance
        for _, d in graph.nodes_iter(data=True):
            d['dist'] = None

        bins = bin_sites.get(graph.graph['id']['tr_name'].split('.')[0], False)
        if bins is False:
            yield graph
        else:
            for _, _, c in bins:
                graph = _bfs_min_dist(graph, c, max_distance)
            yield graph


def get_positive_subgraphs(graph_with_dist, max_dist, split_window):
    """Compute the start and end coordinates of positive subgraphs."""
    bin_centers = [k for k, n in graph_with_dist.node.iteritems()
                   if n['dist'] == 0]
    tr_len = graph_with_dist.graph['id']['tr_len']
    positive_subs = list()
    for bin_center in bin_centers:
        start = max(0, bin_center - max_dist - split_window / 2 + 1)
        end = min(bin_center + max_dist +
                  int(ceil(float(split_window) / 2)) - 1, tr_len)
        positive_subs.append((start, end))
    return positive_subs


def _find_intervals(node_list):
    """Find intervals of continuos points in a node list."""
    intervals = list()
    for k, g in groupby(enumerate(node_list), lambda (i, x): i - x):
        ran = map(itemgetter(1), g)
        if len(ran) > 1:
            start = ran[0]
            end = ran[-1]
            intervals.append((start, end))
    return intervals


def get_border_subgraphs(graph_with_dist, max_dist, split_window):
    """Compute the start and end coordinates of border subgraphs."""
    exact_borders = [k for k, n in graph_with_dist.node.iteritems() if n[
        'dist'] == max_dist]
    putative_borders = [k for k, n in graph_with_dist.node.iteritems(
    ) if max_dist <= n['dist'] < max_dist + split_window / 2]  # check <
    intervals = _find_intervals(putative_borders)
    tr_len = graph_with_dist.graph['id']['tr_len']
    border_subs = list()
    for i1, i2 in intervals:
        if i1 in exact_borders:
            start = max(0, i1 - int(ceil(float(split_window) / 2)))
            end = min(start + max_dist, i2)
            if end - start == max_dist:
                border_subs.append((start, end))
        if i2 in intervals:
            end = min(i2 + split_window / 2, tr_len)
            start = max(i1, end - max_dist)
            if end - start == max_dist:
                border_subs.append((start, end))

    return border_subs


def get_negative_subgraphs(graph_with_dist, max_dist, split_window,
                           negative_ratio=1, random_state=1234):
    """Compute the start and end coordinates of negative subgraphs."""
    # check >=
    putative_negatives = [k for k, n in graph_with_dist.node.iteritems(
    ) if n['dist'] is None or n['dist'] >= max_dist + split_window / 2]
    intervals = _find_intervals(putative_negatives)
    good_centers = list()
    for i1, i2 in intervals:
        if i1 < split_window / 2:
            i1 = split_window / 2
        tr_len = graph_with_dist.graph['id']['tr_len']
        if i2 > tr_len - split_window / 2:
            i2 = tr_len - split_window / 2
        win = i2 - i1 - 2 * max_dist
        center = common.center(i1, i2)
        for _ in range(win):
            good_centers.extend(range(center - win / 2, center + win / 2))
    random.seed(random_state)
    if negative_ratio < len(good_centers):
        good_centers = random.sample(good_centers, negative_ratio)
    negative_subs = [(c - max_dist - split_window / 2 + 1, c +
                      max_dist + split_window / 2 - 1) for c in good_centers]
    return negative_subs


def _subgraph(graph, start, end, **add_params):
    """Generate subgraph from backbone positions plus base-paired nodes.

    Parameters
    ----------
    graph : NetworkX Graph
        Entire graph.

    start : int
        Backbone starting position for the subgraph.

    end : int
        Backbone ending position for the subgraph.

    **add_params : dict
        Dictionary of parameters to add to the 'id' field of the subgraph.

    Returns
    -------
    sub : NetworkX Graph
        Subgraph containing all the backbone nodes between start and end, plus
        all the nodes (and edges of these nodes) connected to the backbone of
        the subgraph through a 'basepair' edge.
    """
    nodes = []
    backbone = np.arange(start, end)
    nodes.extend(backbone)
    for nucl in backbone:
        for neigh in graph.neighbors(nucl):
            if graph.edge[nucl][neigh]['type'] == 'basepair':
                if graph.edge[nucl][neigh].get('nesting', False) is False:
                    nodes.append(neigh)
    nodes = list(set(nodes))
    sub = graph.subgraph(nodes).copy()
    sub.graph['id']['start'] = start
    sub.graph['id']['end'] = end
    sub.graph['id']['length'] = end - start
    for k, v in add_params.iteritems():
        sub.graph['id'][k] = v
    return sub


def train_selector(graphs, bin_sites, max_dist, random_state=1234, **params):
    """Select training subgraphs from folded RNA folded molecules.

    Parameters
    ----------
    graphs : iterable
        Graphs computed with one of the EDeN wrappers for the Vienna RNA
        package.

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
    sub : NetworkX Graph
        Subgraph containing all the backbone nodes between start and end, plus
        all the nodes (and edges of these nodes) connected to the backbone of
        the subgraph through a 'basepair' edge.
    """
    negative_ratio = params.get('negative_ratio', 1)
    split_window = params.get('split_window', 50)

    non_binding = 0
    negative_remainder = 0
    graphs, graphs_ = tee(graphs)

    for g in graphs_:
        bins = bin_sites.get(g.graph['id']['tr_name'].split('.')[0], False)
        if bins is False:
            non_binding += 1
            continue

        # positive
        positive_subs = get_positive_subgraphs(g, max_dist, split_window)
        for start, end in positive_subs:
            yield _subgraph(g, start, end, type='POS')

        # border
        border_subs = get_border_subgraphs(g, max_dist, split_window)
        for start, end in border_subs:
            yield _subgraph(g, start, end, type='BOR')

        # negative
        negative_subs = get_negative_subgraphs(g, max_dist, split_window,
                                               negative_ratio, random_state)
        negative_remainder += max(0, len(positive_subs) *
                                  negative_ratio - len(negative_subs))
        for start, end in negative_subs:
            yield _subgraph(g, start, end, type='NEG')

    # account for missing negative subgraphs
    if negative_remainder > 0 and non_binding > 0:
        new_negative_ratio = int(
            ceil(float(negative_remainder) / non_binding))
        for g in graphs:
            bins = bin_sites.get(g.graph['id']['tr_name'].split('.')[0], False)
            if bins is False and negative_remainder > 1:
                negative_subs = get_negative_subgraphs(g, max_dist,
                                                       split_window,
                                                       new_negative_ratio,
                                                       random_state)
                for start, end in negative_subs:
                    negative_remainder -= 1
                    yield _subgraph(g, start, end, type='NEG')


def split_iterator(graphs, **params):
    """Split graphs.

    Parameters
    ----------
    graphs : iterable
        Graphs computed with one of the EDeN wrappers for the Vienna RNA
        package.

    **params : dict
        Dictionary of pre_processing parameters.

    Returns
    -------
    graph_out : NetworkX Graph
        Subgraph.
    """
    step = params.get('split_step', 1)
    window = params.get('split_window', 50)
    for graph in graphs:
        rna_len = graph.graph['id']['length']
        if rna_len >= window:
            rna_start = graph.graph['id'].get('start', 0)
            rna_end = graph.graph['id'].get('end', rna_len)
            for start in range(rna_start, rna_end - window + 1, step):
                graph_out = _subgraph(graph, start, start + window)
                if graph_out.graph['id']['length'] == window:
                    graph_out.graph['id']['start'] = start
                    graph_out.graph['id']['end'] = graph_out.graph[
                        'id']['start'] + window
                    graph_out.graph['id']['length'] = window
                    graph_out.graph['id']['center'] = common.center(
                        graph_out.graph['id']['start'],
                        graph_out.graph['id']['end'])
                    if graph_out.node[
                            graph_out.graph['id']['center']].get(
                            'dist', False) is not False:
                        graph_out.graph['id']['dist'] = graph_out.node[
                            graph_out.graph['id']['center']]['dist']
                    yield graph_out


def _transform_dictionary(graphs):
    dic = dict()
    for g in graphs:
        dic[g.graph['id']['tr_name'].split('.')[0]] = g
    return dic


# TO DO: figure out the multiprocess, if fails remove n_jobs
def _graph_preprocessor(graphs, which_set, bin_sites=None, max_dist=None,
                        random_state=1234, n_jobs=-1, **params):
    """Preprocess graphs."""
    assert which_set == 'train' or \
        which_set == 'test' or \
        which_set == 'onlyfold', \
        "which_set must be 'train', 'test' or 'onlyfold'."

    # return a dictionary with the folded structures (no splitting).
    if which_set == 'onlyfold':
        return _transform_dictionary(graphs)

    if which_set == 'train':
        graphs = add_distance(graphs, bin_sites)
        graphs = train_selector(
            graphs, bin_sites, max_dist, random_state, **params)

    # graphs = mp_pre_process(graphs,
    #                        pre_processor=split_iterator,
    #                        pre_processor_args=params,
    #                        block_size=100,
    #                        n_jobs=n_jobs)

    graphs = split_iterator(graphs, **params)
    return graphs


def rnafold_preprocessor(iterable, which_set, bin_sites=None, max_dist=None,
                         random_state=1234, n_jobs=-1, **params):
    """Fold sequences with RNAfold and preprocess graphs."""
    graphs = rnafold(iterable)
    graphs = _graph_preprocessor(graphs, which_set, bin_sites, max_dist,
                                 random_state, n_jobs, **params)
    return graphs


def rnaplfold_preprocessor(iterable, which_set, bin_sites=None, max_dist=None,
                           random_state=1234, n_jobs=-1, **params):
    """Fold sequences with RNAplfold and preprocess graphs."""
    graphs = rnaplfold(iterable)

    graphs = _graph_preprocessor(graphs, which_set, bin_sites, max_dist,
                                 random_state, n_jobs, **params)
    return graphs


def store_preprocessor(iterable, which_set, bin_sites=None, max_dist=None,
                       random_state=1234, n_jobs=-1, **params):
    """Retrieve folded sequences from HDF store and preprocess graphs."""
    store_path = params.get('store_path')
    data_manager = HDFDataManager(store_path)
    names = [attr['tr_name'] for attr, _ in iterable]
    graphs = data_manager.retrieve(names)

    graphs = _graph_preprocessor(graphs, which_set, bin_sites, max_dist,
                                 random_state, n_jobs, **params)
    return graphs


def print_stats(iterable):
    """Print number of positive, border, and negative subgraphs."""
    pos = bor = neg = 0
    for g in iterable:
        if g.graph['id']['type'] == 'POS':
            pos += 1
        elif g.graph['id']['type'] == 'BOR':
            bor += 1
        elif g.graph['id']['type'] == 'NEG':
            neg += 1
        else:
            raise Exception("ERROR: unrecognized subsequence type:" +
                            str(g.graph['id']['type']))
    print "POS: %i\nBOR: %i\nNEG: %i\nTOT: %i" %\
        (pos, bor, neg, pos + bor + neg)


def _bfs_at_dist(graph, center, pd):
    """BFS to find the nodes at distance pd from center."""
    if pd == 0:
        nodes_at_dist = [center]
    else:
        nodes_at_dist = list()
        visited = set()
        # q is the queue containing the frontier to be expanded in the BFS
        q = deque()
        q.append(center)
        # the map associates to each vertex id the distance from the center
        dist = {}
        dist[center] = 0
        visited.add(center)
        while len(q) > 0:
            # extract the current vertex
            u = q.popleft()
            d = dist[u] + 1
            if d < pd:
                # iterate over the neighbors of the current vertex
                for v in graph.neighbors(u):
                    if v not in visited:
                        if graph.edge[u][v].get('nesting', False) is False:
                            dist[v] = d
                            visited.add(v)
                            q.append(v)
            elif d == pd:
                for v in graph.neighbors(u):
                    if v not in visited:
                        if graph.edge[u][v].get('nesting', False) is False:
                            nodes_at_dist.append(v)
                            visited.add(v)
    return nodes_at_dist


def vote_aggregator(pred_vals, info, max_dist, full_graphs):
    """Get vote profiles from predicted distances and subgraph info.

    The function assumes that the input contains complete data (predictions
    from all subgraphs) for one or more ENTIRE transcripts.

    Params
    ------
    pred : np.array (dtype : float)
        Predicted regression values, from SGDRegressor.

    info : dict
        Dictionary containing subgraph information for each predicted
        distance in pred. The dictionary has 3 keys 'tr_name', 'tr_len' and
        'center' which is the center of the subsequence in the full
        transcript.

    max_dist : int
        Maximum distance from a binding site to be considered relevant.

    full_graphs : dict
        Dictionary containing the graphs representing the secondary structure
        of the entire transcripts.

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
        tr_name = i['tr_name'].split('.')[0]
        tr_len = i['tr_len']
        center = i['center']
        if tr_name not in votes:
            votes[tr_name] = np.zeros(tr_len)
        if pd < max_dist:
            graph = full_graphs[tr_name]
            nodes_at_pd = _bfs_at_dist(graph, center, pd)
            for n in nodes_at_pd:
                votes[tr_name][n] += pv
    return votes

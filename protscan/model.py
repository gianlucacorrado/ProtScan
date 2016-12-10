"""Regression model with parameter optimization."""

import joblib
import random
import time
import datetime
import copy

import numpy as np
from scipy.sparse import vstack

from itertools import tee
from collections import defaultdict

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score

from eden.util import vectorize, serialize_dict
from eden.util import selection_iterator

import protscan.sequence as seq
from eden.sequence import Vectorizer as SeqVectorizer

import protscan.graph as graph
from eden.graph import Vectorizer as GraphVectorizer

import protscan.common as common
from protscan.util import random_partition_iter, iterator_size
from protscan.util import np_random_permutation
from protscan.util import balanced_fraction, balanced_split
from protscan.util import additive_update

import logging
logger = logging.getLogger(__name__)

__author__ = "Gianluca Corrado, Fabrizio Costa"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


class RegressionModel(object):
    """Regression Model."""

    def __init__(self,
                 mode='sequence',
                 random_state=1234):
        """Constructor.

        Params
        ------
        mode : str
            Values: 'sequence', 'rnafold' or 'rnaplfold'.

        n_jobs : int (default : -1)
            Number of jobs.

        random_state : int (default : 1234)
            Seed for random number generator.
        """
        self.mode = mode

        self.max_dist = None
        self.preprocessor_args = dict()
        self.vectorizer_args = dict()
        self.regressor_args = dict()
        self.smoothing_args = dict()

        if mode == 'sequence':
            self.preprocessor = seq.sequence_preprocessor
            self.vote_aggregator = seq.vote_aggregator
        elif mode == 'rnafold' or mode == 'rnaplfold':
            if mode == 'rnafold':
                self.preprocessor = graph.rnafold_preprocessor
            else:
                self.preprocessor = graph.rnaplfold_preprocessor
            self.vote_aggregator = graph.vote_aggregator
        else:
            raise Exception("Unrecognized mode: %s" % mode)
            exit(1)

        self.regressor = SGDRegressor(shuffle=True,
                                      random_state=random_state)

        # status variables
        self.is_optimized = False
        self.is_fitted = False

    def save(self, model_name):
        """Save model to file."""
        joblib.dump(self, model_name, compress=1, protocol=2)

    def load(self, obj):
        """Load model from file."""
        self.__dict__.update(joblib.load(obj).__dict__)

    def get_mode(self):
        """Return the mode."""
        return self.mode

    def get_parameters(self):
        """Return current parameter setting."""
        text = []
        text.append("-" * 80)
        text.append("max_dist: %s" % self.max_dist)
        text.append("\nPreprocessing:")
        text.append(serialize_dict(self.preprocessor_args))
        text.append("\nVectorizer:")
        text.append(serialize_dict(self.vectorizer_args))
        text.append("\nRegressor:")
        text.append(serialize_dict(self.regressor_args))
        text.append("\nSmoothing:")
        text.append(serialize_dict(self.smoothing_args))
        text.append("-" * 80)
        return '\n'.join(text)

    def get_outer_parameters(self):
        """Return current outer parameter setting."""
        text = []
        text.append("-" * 80)
        text.append("max_dist: %s" % self.max_dist)
        text.append("\nPreprocessing:")
        text.append(serialize_dict(self.preprocessor_args))
        text.append("\nVectorizer:")
        text.append(serialize_dict(self.vectorizer_args))
        text.append("\nRegressor:")
        text.append(serialize_dict(self.regressor_args))
        text.append("-" * 80)
        return '\n'.join(text)

    def get_inner_parameters(self):
        """Return current inner parameter setting."""
        text = []
        text.append("-" * 80)
        text.append("Smoothing:")
        text.append(serialize_dict(self.smoothing_args))
        text.append("-" * 80)
        return '\n'.join(text)

    def get_supervised_data(self, preprocessed, bin_sites,
                            active_learning=False, random_state=1234,
                            n_jobs=-1):
        """Compute the feature matrix and the regression values."""
        preprocessed, preprocessed_ = tee(preprocessed)
        if self.mode == 'sequence':
            dists = [attr['dist'] for attr, _ in preprocessed_]
        else:
            dists = [g.graph['id']['dist'] for g in preprocessed_]
        vals = np.array([common.dist_to_val(d, self.max_dist) for d in dists])
        if self.mode == 'sequence':
            self.vectorizer = SeqVectorizer(auto_weights=True,
                                            **self.vectorizer_args)
        else:
            self.vectorizer = GraphVectorizer(auto_weights=True,
                                              **self.vectorizer_args)
        matrix = vectorize(preprocessed, vectorizer=self.vectorizer,
                           block_size=400, n_jobs=n_jobs)
        return matrix, vals

    def get_predict_data(self, preprocessed, n_jobs=-1):
        """Compute the feature matrix and extract the subseq info."""
        def _subdict(dic):
            subdict = dict((k, dic[k]) for k in [
                           'tr_name', 'center', 'tr_len'] if k in dic)
            return subdict

        preprocessed, preprocessed_ = tee(preprocessed)
        if self.mode == 'sequence':
            info = [_subdict(attr) for attr, _ in preprocessed_]
        else:
            info = [_subdict(g.graph['id']) for g in preprocessed_]

        if self.mode == 'sequence':
            self.vectorizer = SeqVectorizer(auto_weights=True,
                                            **self.vectorizer_args)
        else:
            self.vectorizer = GraphVectorizer(auto_weights=True,
                                              **self.vectorizer_args)
        matrix = vectorize(preprocessed, vectorizer=self.vectorizer,
                           block_size=400, n_jobs=n_jobs)
        return matrix, info

    def _fit(self, sequences, bin_sites, fit_batch_size=500,
             max_splits=100000, active_learning=False,
             random_state=1234, n_jobs=-1):
        """Fit the regressor (using partial fit)."""
        self.regressor.set_params(**self.regressor_args)
        sequences, sequences_ = tee(sequences)
        size = iterator_size(sequences_)
        n_batches = max(1, size / fit_batch_size)
        batches = random_partition_iter(sequences, n_batches, random_state)
        # it might differ from the requested one
        n_batches = len(batches)
        logger.debug("Fitting (%d batch%s)..." %
                     (n_batches, "es" * (n_batches > 1)))
        for i, batch in enumerate(batches):
            start_time = time.time()
            preprocessed = self.preprocessor(batch,
                                             which_set='train',
                                             bin_sites=bin_sites,
                                             max_dist=self.max_dist,
                                             random_state=random_state,
                                             **self.preprocessor_args)
            preprocessed, preprocessed_ = tee(preprocessed)
            if active_learning is True:
                batch_split_size = iterator_size(preprocessed_)
                n_parts = max(1, batch_split_size / max_splits)
                parts = random_partition_iter(
                    preprocessed, n_parts, random_state)
                for j, part in enumerate(parts):
                    # features of all the splits (with regression values)
                    full_matrix, full_vals = self.get_supervised_data(
                        part, bin_sites, random_state, n_jobs)

                    # random selection of negative examples
                    pos_matrix = full_matrix[full_vals > 0.0]
                    pos_vals = full_vals[full_vals > 0.0]
                    neg_matrix = full_matrix[full_vals == 0.0]
                    neg_vals = full_vals[full_vals == 0.0]

                    neg_matrix, neg_vals = np_random_permutation(
                        neg_matrix,
                        neg_vals,
                        random_state)

                    n_pos = len(pos_vals)
                    negative_ratio = self.preprocessor_args.get(
                        'negative_ratio')
                    n_neg = int(n_pos * negative_ratio)
                    matrix = vstack(
                        (pos_matrix, neg_matrix[:n_neg]), format='csr')
                    vals = np.concatenate((pos_vals, neg_vals[:n_neg]))

                    # fit a temporary regressor with random negatives
                    matrix, vals = np_random_permutation(matrix,
                                                         vals,
                                                         random_state)
                    temp_regressor = copy.deepcopy(self.regressor)
                    if i == 0 and j == 0:
                        # initialize regressor weights
                        temp_regressor.fit(matrix, vals)
                    else:
                        temp_regressor.partial_fit(matrix, vals)

                    # find negative examples with the highest predicted values
                    pred_neg_vals = temp_regressor.predict(neg_matrix)
                    order = np.argsort(pred_neg_vals)[::-1]
                    neg_matrix = neg_matrix[order]
                    neg_vals = neg_vals[order]
                    matrix = vstack(
                        (pos_matrix, neg_matrix[:n_neg]), format='csr')
                    vals = np.concatenate((pos_vals, neg_vals[:n_neg]))

                    # fit a the regressor using most the confusing negatives
                    matrix, vals = np_random_permutation(matrix,
                                                         vals,
                                                         random_state)
                    if i == 0 and j == 0:
                        # initialize regressor weights
                        self.regressor.fit(matrix, vals)
                    else:
                        self.regressor.partial_fit(matrix, vals)

                delta_time = datetime.timedelta(
                    seconds=(time.time() - start_time))
                logger.debug("\tbatch %d/%d, elapsed time: %s" %
                             (i + 1, n_batches, str(delta_time)))
            else:
                # random selection of negative splits
                pos_idx, neg_idx = list(), list()
                if self.mode == 'sequence':
                    for j, (attr, _) in enumerate(preprocessed_):
                        if attr['type'] == 'POS':
                            pos_idx.append(j)
                        else:
                            neg_idx.append(j)
                else:
                    for j, g in enumerate(preprocessed_):
                        if g.graph['id']['type'] == 'POS':
                            pos_idx.append(j)
                        else:
                            neg_idx.append(j)

                np.random.seed(random_state)
                np.random.shuffle(neg_idx)
                n_pos = len(pos_idx)
                negative_ratio = self.preprocessor_args.get('negative_ratio')
                n_neg = int(n_pos * negative_ratio)
                idx = np.concatenate((pos_idx, neg_idx[:n_neg]))
                selected = selection_iterator(preprocessed, idx)

                selected, selected_ = tee(selected)
                batch_split_size = iterator_size(selected_)
                n_parts = max(1, batch_split_size / max_splits)
                parts = random_partition_iter(selected, n_parts, random_state)

                for j, part in enumerate(parts):
                    # features of selected splits (with regression values)
                    matrix, vals = self.get_supervised_data(
                        part, bin_sites, random_state, n_jobs)

                    # fit the regressor with random negatives
                    matrix, vals = np_random_permutation(matrix,
                                                         vals,
                                                         random_state)
                    if i == 0 and j == 0:
                        # initialize regressor weights
                        self.regressor.fit(matrix, vals)
                    else:
                        self.regressor.partial_fit(matrix, vals)

                delta_time = datetime.timedelta(
                    seconds=(time.time() - start_time))
                logger.debug("\tbatch %d/%d, elapsed time: %s" %
                             (i + 1, n_batches, str(delta_time)))
        logger.debug("Done!")

    def fit(self, sequences, bin_sites, model_name, fit_batch_size=500,
            max_splits=100000, active_learning=False, random_state=1234,
            n_jobs=-1):
        """Fit the regressor (using partial fit), and save the model."""
        self._fit(sequences, bin_sites, fit_batch_size, max_splits,
                  active_learning, random_state, n_jobs)
        self.is_fitted = True
        self.save(model_name)

    def vote(self, sequences, pre_batch_size=200, max_splits=100000,
             random_state=1234, n_jobs=-1):
        """Collect the votes for the binding profiles."""
        sequences, sequences_ = tee(sequences)
        size = iterator_size(sequences_)
        n_batches = max(1, size / pre_batch_size)
        batches = random_partition_iter(sequences, n_batches, random_state)
        # it might differ from the requested one
        n_batches = len(batches)
        logger.debug("Predicting (%d batch%s)..." %
                     (n_batches, "es" * (n_batches > 1)))
        votes = dict()
        for i, batch in enumerate(batches):
            start_time = time.time()
            if self.mode == 'sequence':
                preprocessed = self.preprocessor(batch,
                                                 which_set='test',
                                                 **self.preprocessor_args)
            else:
                full_graphs, preprocessed = self.preprocessor(
                    batch,
                    which_set='test',
                    **self.preprocessor_args)

            preprocessed, preprocessed_ = tee(preprocessed)
            batch_split_size = iterator_size(preprocessed_)
            n_parts = max(1, batch_split_size / max_splits)
            parts = random_partition_iter(preprocessed, n_parts, random_state)

            for part in parts:
                matrix, info = self.get_predict_data(part, n_jobs)
                pred_vals = self.regressor.predict(matrix)
                part_votes = self.vote_aggregator(pred_vals, info,
                                                  self.max_dist)
                additive_update(votes, part_votes)

            delta_time = datetime.timedelta(
                seconds=(time.time() - start_time))
            logger.debug("\tbatch %d/%d, elapsed time: %s" %
                         (i + 1, n_batches, str(delta_time)))
        logger.debug("Done!")
        return votes

    def smooth(self, votes):
        """Smooth the votes and obtain binding profiles."""
        profiles = common.smooth(votes, **self.smoothing_args)
        return profiles

    def predict(self, sequences, pre_batch_size=200, max_splits=100000,
                random_state=1234, n_jobs=-1):
        """Predict binding profiles for a set of sequences."""
        votes = self.vote(sequences, pre_batch_size,
                          max_splits, random_state, n_jobs)
        profiles = self.smooth(votes)
        return profiles

    def cross_vote(self, sequences, bin_sites, fit_batch_size=500,
                   pre_batch_size=200, max_splits=100000,
                   active_learning=False, random_state=1234, n_jobs=-1):
        """2-fold cross fit and vote."""
        votes = dict()
        part1, part2 = balanced_split(sequences, bin_sites, n_splits=2,
                                      random_state=random_state)

        part1, part1_ = tee(part1)
        part2, part2_ = tee(part2)

        # fold 1
        logger.debug("Fold 1")
        tr, te = part1, part2
        self._fit(tr, bin_sites, fit_batch_size, max_splits, active_learning,
                  random_state, n_jobs)
        part_votes = self.vote(
            te, pre_batch_size, max_splits, random_state, n_jobs)
        votes.update(part_votes)

        # fold 2
        logger.debug("Fold 2")
        tr, te = part2_, part1_
        self._fit(tr, bin_sites, fit_batch_size, max_splits, active_learning,
                  random_state, n_jobs)
        part_votes = self.vote(
            te, pre_batch_size, max_splits, random_state, n_jobs)
        votes.update(part_votes)
        return votes

    def cross_predict(self, sequences, bin_sites, fit_batch_size=500,
                      pre_batch_size=200, max_splits=100000,
                      active_learning=False, random_state=1234, n_jobs=-1):
        """2-fold cross fit and predict."""
        logger.debug("Model parameters:")
        logger.debug(self.get_parameters())
        votes = self.cross_vote(sequences, bin_sites, fit_batch_size,
                                pre_batch_size, max_splits, active_learning,
                                random_state, n_jobs)
        profiles = self.smooth(votes)
        return profiles

    def score(self, profiles, bin_sites):
        """Compute AUC ROC from predictions."""
        app_profiles = list()
        app_true_vals = list()
        for k, profile in profiles.iteritems():
            app_profiles.append(profile)
            true_vals = np.zeros(len(profile))
            bins = bin_sites.get(k, False)
            if bins is not False:
                for s, e, _ in bins:
                    true_vals[s:e] = 1
            app_true_vals.append(true_vals)
        vec_profiles = np.concatenate(app_profiles)
        vec_true_vals = np.concatenate(app_true_vals)
        roc_auc = roc_auc_score(vec_true_vals, vec_profiles)
        return roc_auc

    def score_from_votes(self, votes, bin_sites):
        """Smooth and compute AUC ROC."""
        profiles = self.smooth(votes)
        roc_auc = self.score(profiles, bin_sites)
        return roc_auc

    def optimize(self,
                 sequences,
                 bin_sites,
                 model_name,
                 n_iter=20,
                 n_smoothing_iter=20,
                 opt_fraction=1.0,
                 opt_svm=False,
                 max_dist_vals=list(),
                 preprocessor_params=dict(),
                 vectorizer_params=dict(),
                 regressor_params=dict(),
                 smoothing_params=dict(),
                 two_steps_opt=False,
                 fit_batch_size=500,
                 pre_batch_size=200,
                 max_splits=100000,
                 active_learning=False,
                 fit_with_opt_params=False,
                 max_total_time=-1,
                 random_state=1234,
                 n_jobs=-1):
        """Optimize model params and fit using with the optimal params."""
        def _get_parameters_range():
            text = []
            text.append('\n\n\tParameters range:')
            text.append('\nmax_dists: %s' % max_dist_vals)
            text.append('\nPreprocessor:')
            text.append(serialize_dict(preprocessor_params))
            text.append('\nVectorizer:')
            text.append(serialize_dict(vectorizer_params))
            text.append('\nRegressor:')
            text.append(serialize_dict(regressor_params))
            text.append('\nSmoothing:')
            text.append(serialize_dict(smoothing_params))
            return '\n'.join(text)

        logger.debug(_get_parameters_range())

        # init
        n_failures = 0

        best_score_ = 0.
        best_max_dist_ = None
        best_preprocessor_args_ = dict()
        best_vectorizer_args_ = dict()
        best_regressor_args_ = dict()
        best_smoothing_args_ = dict()

        best_max_dist_vals_ = list()
        best_preprocessor_params_ = defaultdict(list)
        best_vectorizer_params_ = defaultdict(list)
        best_regressor_params_ = defaultdict(list)
        best_smoothing_params_ = defaultdict(list)

        start = time.time()

        if opt_fraction > 0.0 and opt_fraction < 1.0:
            sequences, sequences_ = tee(sequences)
            opt_sequences = balanced_fraction(sequences_, bin_sites,
                                              opt_fraction)
        elif opt_fraction == 1.0:
            opt_sequences, sequences = tee(sequences)
        else:
            logger.debug("Error", exc_info=True)
            delta_time = datetime.timedelta(
                seconds=(time.time() - start))
            text = []
            text.append(
                "\nopt_fraction must be > 0.0 <= 1.0")
            text.append("quitting.")
            logger.debug('\n'.join(text))
            exit(1)

        if n_iter == 1:
            logger.debug("n_iter is 1: switching to default parameters")
            self.max_dist = max_dist_vals[0]
            self.preprocessor_args = self._default(preprocessor_params)
            self.vectorizer_args = self._default(vectorizer_params)
            self.regressor_args = self._default(regressor_params)
            self.smoothing_args = self._default(smoothing_params)
        else:
            # main iteration
            for i in range(n_iter):
                logger.debug("\nIteration %d" % (i + 1))
                if max_total_time != -1:
                    if time.time() - start > max_total_time:
                        delta_time = datetime.timedelta(
                            seconds=(time.time() - start))
                        logger.warning("Reached max time: %s" %
                                       (str(delta_time)))
                        break

                # after (n_iter / 2) iterations, replace the parameter lists
                # with only those values that have been found to increase
                # the performance
                if i == int(n_iter / 2) and two_steps_opt is True:
                    if len(best_max_dist_vals_) > 0:
                        max_dist_vals = best_max_dist_vals_
                    if len(best_preprocessor_params_) > 0:
                        preprocessor_params = dict(best_preprocessor_params_)
                    if len(best_vectorizer_params_) > 0:
                        vectorizer_params = dict(best_vectorizer_params_)
                    if len(best_regressor_params_) > 0:
                        regressor_params = dict(best_regressor_params_)
                    if len(best_smoothing_params_) > 0:
                        smoothing_params = dict(best_smoothing_params_)
                    # logger.debug(_get_parameters_range())

                    if len(max_dist_vals) == 1 and \
                            len(preprocessor_params) == 1 and \
                            len(vectorizer_params) == 1 and \
                            len(regressor_params) == 1 and \
                            len(smoothing_params) == 1:
                        logger.debug(
                            "Optimal params range is singular, bailing out.")
                        break
                # during the first iteration, select the default parameters
                if i == 0:
                    self.max_dist = max_dist_vals[0]
                    self.preprocessor_args = self._default(preprocessor_params)
                    self.vectorizer_args = self._default(vectorizer_params)
                    self.regressor_args = self._default(regressor_params)
                # for the other iterations, sample the parameters
                else:
                    self.max_dist = random.choice(max_dist_vals)
                    self.preprocessor_args = self._sample(
                        preprocessor_params, random_state=random_state + i)
                    self.vectorizer_args = self._sample(
                        vectorizer_params, random_state=random_state + i)

                    if opt_svm:
                        self.regressor_args = self._sample(
                            regressor_params, random_state=random_state + i)
                    else:
                        self.regressor_args = self._default(regressor_params)

                logger.debug("Trying outer parameters:")
                logger.debug(self.get_outer_parameters())

                try:
                    opt_sequences, opt_sequences_ = tee(opt_sequences)
                    votes = self.cross_vote(opt_sequences_, bin_sites,
                                            fit_batch_size, pre_batch_size,
                                            max_splits, active_learning,
                                            random_state, n_jobs)
                except Exception as e:
                    logger.debug("Exception", exc_info=True)
                    delta_time = datetime.timedelta(
                        seconds=(time.time() - start))
                    text = []
                    text.append(
                        "\nFailed optimization iteration:")
                    text.append("%d/%d (at %.1f sec; %s)" %
                                (i + 1, n_iter, time.time() - start,
                                    str(delta_time)))
                    text.append(e.__doc__)
                    text.append(e.message)
                    text.append("Failed with the following setting:")
                    text.append(self.get_outer_parameters())
                    text.append("...continuing")
                    logger.debug('\n'.join(text))
                    n_failures += 1
                    continue

                for smoothing_i in range(n_smoothing_iter):
                    # during the first iteration, select the default parameters
                    if i == 0 and smoothing_i == 0:
                        self.smoothing_args = self._default(smoothing_params)
                    # for the other iterations, sample the parameters
                    else:
                        self.smoothing_args = self._sample(
                            smoothing_params, random_state=random_state + i +
                            smoothing_i)

                    logger.debug("Trying smoothing parameters:")
                    logger.debug(self.get_inner_parameters())

                    try:
                        score = self.score_from_votes(votes, bin_sites)
                    except Exception as e:
                        logger.debug("Exception", exc_info=True)
                        delta_time = datetime.timedelta(
                            seconds=(time.time() - start))
                        text = []
                        text.append(
                            "\nFailed smoothing optimization iteration:")
                        text.append("(%d/%d) %d/%d (at %.1f sec; %s)" %
                                    (smoothing_i + 1, n_smoothing_iter, i + 1,
                                        n_iter, time.time() - start,
                                        str(delta_time)))
                        text.append(e.__doc__)
                        text.append(e.message)
                        text.append("Failed with the following setting:")
                        text.append(self.get_parameters())
                        text.append("...continuing")
                        logger.debug('\n'.join(text))
                        n_failures += 1
                        continue

                    if best_score_ < score:
                        # save the model, with the current best params
                        self.save(model_name)
                        # remember the current param configuration
                        best_score_ = score
                        best_max_dist_ = self.max_dist
                        best_preprocessor_args_ = copy.deepcopy(
                            self.preprocessor_args)
                        best_vectorizer_args_ = copy.deepcopy(
                            self.vectorizer_args)
                        best_regressor_args_ = copy.deepcopy(
                            self.regressor_args)
                        best_smoothing_args_ = copy.deepcopy(
                            self.smoothing_args)
                        # add parameter to list of best parameters
                        best_max_dist_vals_.append(self.max_dist)
                        for key in self.preprocessor_args:
                            best_preprocessor_params_[key].append(
                                self.preprocessor_args[key])
                        for key in self.vectorizer_args:
                            best_vectorizer_params_[key].append(
                                self.vectorizer_args[key])
                        for key in self.regressor_args:
                            best_regressor_params_[key].append(
                                self.regressor_args[key])
                        for key in self.smoothing_args:
                            best_smoothing_params_[key].append(
                                self.smoothing_args[key])

                        delta_time = datetime.timedelta(
                            seconds=(time.time() - start))
                        text = []
                        text.append("-" * 80)
                        text.append("*" * 80)
                        text.append("New best solution found:")
                        text.append(
                            "\tIteration: %d/%d (after %.1f sec; %s)" %
                            (i + 1, n_iter, time.time() - start,
                             str(delta_time)))
                        text.append(
                            "\tSmoothing iteration: %d/%d" %
                            (smoothing_i + 1, n_smoothing_iter))
                        text.append(self.get_parameters())
                        text.append(
                            "Best score (AUC ROC): %.3f" % best_score_)
                        text.append("*" * 80)
                        text.append("-" * 80)
                        logger.info('\n'.join(text))

            # store the best param configuration and save the model
            self.max_dist = best_max_dist_
            self.preprocessor_args = copy.deepcopy(best_preprocessor_args_)
            self.vectorizer_args = copy.deepcopy(best_vectorizer_args_)
            self.regressor_args = copy.deepcopy(best_regressor_args_)
            self.smoothing_args = copy.deepcopy(best_smoothing_args_)
            self.save(model_name)

        # save final model.
        if n_failures < n_iter * n_smoothing_iter:
            self.is_optimized = True
            if fit_with_opt_params is True:
                self._fit(sequences, bin_sites, fit_batch_size, max_splits,
                          active_learning, random_state, n_jobs)
                self.is_fitted = True
                self.save(model_name)
                logger.info("Model fitted using best param configuration.")
                logger.info(self.get_parameters())
            else:
                self.save(model_name)
                logger.info("Best params configuration saved:")
                logger.info(self.get_parameters())
                logger.info("Model requires fit.")

        else:
            logger.warning(
                "ERROR: no iteration has produced any viable solution.")
            exit(1)

    def _sample(self, parameters, random_state):
        """Sample parameters at random."""
        random.seed(random_state)
        parameters_sample = dict()
        for parameter in parameters:
            values = parameters[parameter]
            value = random.choice(values)
            parameters_sample[parameter] = value
        return parameters_sample

    def _default(self, parameters):
        """Select default (position 0) parameters."""
        parameters_sample = dict()
        for parameter in parameters:
            values = parameters[parameter]
            value = values[0]
            parameters_sample[parameter] = value
        return parameters_sample

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

import logging
import operator
from collections import Iterable

import numpy
from six.moves import zip_longest

try:
    basestring = basestring
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str, bytes)

from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
from sklearn.base import BaseEstimator, ClassifierMixin

from .word2vec import LabeledWord2Vec, score_document_labeled_cbow

__author__ = 'Giacomo Berardi <giacbrd.com>'

logger = logging.getLogger(__name__)


# TODO model based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score
# TODO model based on https://github.com/salestock/fastText.py


class BaseClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self._classifier = None
        self._label_set = None
        self._label_type = None

    @classmethod
    def _target_list(cls, targets):
        return targets if isinstance(targets, Iterable) and not isinstance(targets, basestring) else [targets]

    def _build_label_info(self, y):
        self._label_set = frozenset(target for targets in y for target in self._target_list(targets))
        self._label_is_num = isinstance(next(iter(self._label_set)), (int, float, complex))


class GensimFastText(BaseClassifier):
    """
    A supervised learning model based on the fastText algorithm [1]_ and written in Python.
    The core code, as this documentation, is copied from `Gensim <https://radimrehurek.com/gensim>`_,
    it takes advantage of its optimizations and support.
    The parameter names are equivalent to the ones in the original fasText implementation (https://github.com/facebookresearch/fastText).

    For now it only uses Hierarchical Softmax for output computation, and it is obviously limited to the CBOW method.

    `size` is the dimensionality of the feature vectors.

    `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

    `random_state` = for the random number generator. Initial vectors for each
    word are seeded with a hash of the concatenation of word + str(seed).
    Note that for a fully deterministically-reproducible run, you must also limit the model to
    a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
    3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
    environment variable to control hash randomization.)

    `min_count` = ignore all words with total frequency lower than this.

    `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
    words than this, then prune the infrequent ones. Every 10 million word types
    need about 1GB of RAM. Set to `None` for no limit (default).

    `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
        default is 1e-3, useful range is (0, 1e-5).

    `workers` = use this many worker threads to train the model (=faster training with multicore machines).

    `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
    Only applies when cbow is used.

    `hashfxn` = hash function to use to randomly initialize weights, for increased
    training reproducibility. Default is Python's rudimentary built in hash function.

    `max_iter` = number of iterations (epochs) over the corpus. Default is 5.

    `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
    in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
    Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
    returns either `utils.RULE_DISCARD`, `utils.RULE_KEEP` or `utils.RULE_DEFAULT`.
    Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
    of the model.

    `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
    assigning word indexes.

    `batch_words` = target size (in words) for batches of examples passed to worker threads (and
    thus cython routines). Default is 10000. (Larger batches will be passed if individual
    texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

    `pre_trained` can be set with a ``Word2Vec`` object in order to set pre-trained word vectors and vocabularies.
    Use ``partial_fit`` method to learn a supervised model starting from the pre-trained one.

    .. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

    """

    def __init__(self, size=200, alpha=0.05, min_count=5, max_vocab_size=None, sample=1e-3, workers=3,
                 min_alpha=0.0001, cbow_mean=1, hashfxn=hash, null_word=0, trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, max_iter=5, random_state=1, pre_trained=None):
        # FIXME logging configuration must be project wise, rewrite this condition
        super(GensimFastText, self).__init__()
        self.set_params(
            size=size,
            alpha=alpha,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            sample=sample,
            workers=workers,
            min_alpha=min_alpha,
            cbow_mean=cbow_mean,
            hashfxn=hashfxn,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            batch_words=batch_words,
            max_iter=max_iter,
            random_state=random_state
        )
        params = self.get_params()
        # Convert name conventions from Scikit-learn to Gensim
        params['iter'] = params['max_iter']
        del params['max_iter']
        params['seed'] = params['random_state']
        del params['random_state']
        del params['pre_trained']
        self._classifier = LabeledWord2Vec(**params)
        if pre_trained is not None:
            self._classifier.reset_from(pre_trained)

    @classmethod
    def _data_iter(cls, documents, y):

        class DocIter(object):
            def __init__(self, documents, y):
                self.y = y
                self.documents = documents

            def __iter__(self):
                for sample, targets in zip_longest(self.documents, self.y):
                    targets = cls._target_list(targets)
                    yield (sample, targets)

        return DocIter(documents, y)

    @property
    def classifier(self):
        """
        The word embeddings model
        :return: An instance of ``LabeledWord2Vec``, a CBOW model in wich input vectors are words, output vectors are labels.
        """
        return self._classifier

    def fit(self, documents, y=None, **fit_params):
        """
        Train the supervised model with a labeled training set
        :param documents: Iterator over lists of words
        :param y: Iterator over lists or single labels, document target values
        :param fit_params: Nothing for now
        :return:
        """
        # TODO if y=None just learn the word vectors
        self._build_label_info(y)
        if not self._classifier.vocab:
            self._classifier.build_vocab(documents, self._label_set, trim_rule=self.trim_rule)
        self._classifier.train(self._data_iter(documents, y))

    def partial_fit(self, documents, y):
        """
        Train more data over the already trained model.
        :param documents: Iterator over lists of words
        :param y: Iterator over lists or single labels, document target values
        """
        size = sum(1 for _ in self._data_iter(documents, y))
        self._classifier.train(self._data_iter(documents, y), total_examples=size)

    def _iter_predict(self, documents):
        # FIXME instead of iterate on documents, it would be ideal to compute a single operation with the sub-matrix of documents
        for doc in documents:
            result = [(label, score_document_labeled_cbow(self._classifier, document=doc, label=label)) for
                      label in self._label_set]
            result.sort(key=operator.itemgetter(1), reverse=True)
            yield result

    def predict_proba(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, a list of tuples with labels and their probabilities, which should sum to one for each prediction
        """
        return list(self._iter_predict(documents))

    def decision_function(self, documents):
        return self.predict_proba(documents)

    def predict(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, the one most probable label (i.e. the classification)
        """
        # FIXME it only returns the most probable class, so it is not multi-label (even if the training is)
        return [predictions[0][0] for predictions in self._iter_predict(documents)]

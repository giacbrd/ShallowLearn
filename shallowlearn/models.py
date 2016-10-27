#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

import io
import logging
import operator
import tempfile
from collections import Iterable
from numbers import Number

import fasttext
from gensim.utils import to_unicode
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


class BaseClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self._classifier = None
        self._label_set = None
        self._label_is_num = None

    @classmethod
    def _target_list(cls, targets):
        return targets if isinstance(targets, Iterable) and not isinstance(targets, basestring) else [targets]

    def _build_label_info(self, y):
        self._label_set = frozenset(target for targets in y for target in self._target_list(targets))
        self._label_is_num = isinstance(next(iter(self._label_set)), (int, float, complex, Number))

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


class GensimFastText(BaseClassifier):
    """
    A supervised learning model based on the fastText algorithm [1]_ and written in Python.
    The core code, as this documentation, is copied from `Gensim <https://radimrehurek.com/gensim>`_,
    it takes advantage of its optimizations and support.
    The parameter names are a mix of the ones in the original fasText implementation
    (https://github.com/facebookresearch/fastText) and the Gensim ones.

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

    `loss` = one value in {ns, hs, softmax}. If "ns" is selected negative sampling will be used
    as loss function, together with the parameter `negative`. With "hs" hierarchical softmax will be used,
    while with "softmax" (default) the sandard softmax function (the other two are "approximations")

    `negative` = if > 0, negative sampling will be used, the int for negative
    specifies how many "noise words" should be drawn (usually between 5-20).
    Default is 5. If set to 0, no negative samping is used. It works only if `loss = ns`

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

    def __init__(self, size=200, alpha=0.05, min_count=5, max_vocab_size=None, sample=1e-3, loss='softmax', negative=5,
                 workers=3, min_alpha=0.0001, cbow_mean=1, hashfxn=hash, null_word=0, trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, max_iter=5, random_state=1, pre_trained=None):
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
            random_state=random_state,
            loss=loss,
            negative=negative
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
            if hasattr(pre_trained, 'lvocab'):
                self._build_label_info(pre_trained.lvocab.keys())
            self.set_params(
                size=pre_trained.layer1_size,
                alpha=pre_trained.alpha,
                min_count=pre_trained.min_count,
                max_vocab_size=pre_trained.max_vocab_size,
                sample=pre_trained.sample,
                workers=pre_trained.workers,
                min_alpha=pre_trained.min_alpha,
                cbow_mean=pre_trained.cbow_mean,
                hashfxn=pre_trained.hashfxn,
                null_word=pre_trained.null_word,
                sorted_vocab=pre_trained.sorted_vocab,
                batch_words=pre_trained.batch_words,
                max_iter=pre_trained.iter,
                random_state=pre_trained.seed,
                loss='softmax' if pre_trained.softmax else ('hs' if pre_trained.hs else 'ns'),
                negative=pre_trained.negative
            )

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
        # TODO if y=None learn a one-class classifier
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


class FastText(BaseClassifier):
    """
    A wrapper class for Scikit-learn on the `supervised model of fastText.py
    <https://github.com/salestock/fastText.py#supervised-model>`_ with the same parameter names.

    `lr` = learning rate [0.1]

    `lr_update_rate` = change the rate of updates for the learning rate [100]

    `dim` = size of word vectors [100]

    `ws` = size of the context window [5]

    `epoch` = number of epochs [5]

    `min_count` = minimal number of word occurences [1]

    `neg` = number of negatives sampled [5]

    `word_ngrams` = max length of word ngram [1]

    `loss` = loss function {ns, hs, softmax} [softmax]

    `bucket` = number of buckets [0]

    `minn` = min length of char ngram [0]

    `maxn` = max length of char ngram [0]

    `thread` = number of threads [12]

    `t` = sampling threshold [0.0001]

    `silent` = disable the log output from the C++ extension [1]

    `encoding` = specify input_file encoding [utf-8]
    """

    def __init__(self, lr=0.1, lr_update_rate=100, dim=100, ws=5, epoch=5, min_count=1, neg=5, word_ngrams=1,
                 loss='softmax', bucket=0, minn=0, maxn=0, thread=12, t=0.0001, silent=1, encoding='utf-8'):
        super(FastText, self).__init__()
        self.set_params(
            lr=lr, lr_update_rate=lr_update_rate, dim=dim, ws=ws, epoch=epoch, min_count=min_count, neg=neg,
            word_ngrams=word_ngrams, loss=loss, bucket=bucket, minn=minn, maxn=maxn, thread=thread, t=t, silent=silent,
            encoding=encoding
        )
        self.label_prefix = '__label__'
        self._classifier = None

    @property
    def classifier(self):
        """
        The fastText.py supervised model
        :return: An instance of ``fasttext.supervised``
        """
        return self._classifier

    def fit(self, documents, y, **fit_params):
        """
        Train the supervised model with a labeled training set
        :param documents: Iterator over lists of words
        :param y: Iterator over lists or single labels, document target values
        :param fit_params: Nothing for now
        :return:
        """
        self._build_label_info(y)
        with tempfile.NamedTemporaryFile() as train_temp:
            with io.open(train_temp.name, 'w', encoding=self.encoding) as dataset:
                for x, y in self._data_iter(documents, y):
                    if x and y:
                        targets = ' '.join('%s%s' % (self.label_prefix, label) for label in y)
                        sample = ' '.join(x)
                        dataset.write(to_unicode(targets + ' ' + sample + '\n'))
            self.fit_file(train_temp.name)

    def fit_file(self, train_path, output_path=None, label_prefix=None):
        """
        Train the model from a training set file
        :param train_path: Txt file of documents, with one or more `label_prefix`+`category`
            (default `label_prefix` value is "__label__") at the beginning of each line
        followed by the document tokens
        :param output_path: Path where to save the model, a `.bin` extension will be appended
        :param label_prefix:
        """

        def train_classifier(output):
            self._classifier = fasttext.supervised(input_file=train_path, output=output,
                                                   label_prefix=label_prefix or self.label_prefix, **self.get_params())

        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix='.bin') as model_temp:
                train_classifier(model_temp.name[:-4])
        else:
            train_classifier(output_path)

    def predict_proba(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, a list of tuples with labels and their probabilities, which should sum to one for each prediction
        """
        result = self._classifier.predict_proba(iter(' '.join(d) for d in documents), len(self._label_set))
        uniform = 1. / len(self._label_set)
        result = [[(l, uniform) for l in self._label_set] if not any(r) else r for r in result]
        return result

    def decision_function(self, documents):
        return self.predict_proba(documents)

    def predict(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, the one most probable label (i.e. the classification)
        """
        return [((float(pred[0]) if self._label_is_num else pred[0])
                 if pred else None) for pred in self._classifier.predict(iter(' '.join(d) for d in documents), 1)]

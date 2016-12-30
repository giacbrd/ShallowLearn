#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

import io
import logging
import operator
import shutil
import tempfile
from collections import Iterable
from numbers import Number

import fasttext
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
from gensim.utils import to_unicode
from six.moves import zip_longest
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import argument_alternatives, basestring
from .word2vec import LabeledWord2Vec, score_document_labeled_cbow

__author__ = 'Giacomo Berardi <giacbrd.com>'

logger = logging.getLogger(__name__)

CLASSIFIER_FILE_SUFFIX = '.CLF'


class BaseClassifier(ClassifierMixin, BaseEstimator, gensim.utils.SaveLoad):
    def __init__(self):
        self._classifier = None
        self._label_set = None
        self._label_count = None
        self._label_is_num = None
        self.classes_ = None

    @classmethod
    def _target_list(cls, targets):
        return targets if isinstance(targets, Iterable) and not isinstance(targets, basestring) else [targets]

    def _build_label_info(self, y, overwrite=False):
        label_set = set(target for targets in y for target in self._target_list(targets))
        if self._label_set is None or overwrite:
            self._label_set = label_set
        else:
            self._label_set.update(label_set)
        self.classes_ = list(self._label_set)
        self._label_count = len(self._label_set)
        self._label_is_num = isinstance(next(iter(self._label_set)), (int, float, complex, Number))

    def _extract_prediction(self, prediction):
        pred_map = dict(prediction)
        return tuple(pred_map[label] for label in self.classes_)

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

    `size` or `dim` is the dimensionality of the feature vectors.

    `alpha` or `lr` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

    `seed` or `random_state` = for the random number generator. Initial vectors for each
    word are seeded with a hash of the concatenation of word + str(seed).
    Note that for a fully deterministically-reproducible run, you must also limit the model to
    a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
    3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
    environment variable to control hash randomization.)

    `min_count` = ignore all words with total frequency lower than this.

    `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
    words than this, then prune the infrequent ones. Every 10 million word types
    need about 1GB of RAM. Set to `None` for no limit (default).

    `sample` or `t` = threshold for configuring which higher-frequency words are randomly downsampled;
        default is 1e-3, useful range is (0, 1e-5).

    `loss` = one value in {ns, hs, softmax}. If "ns" is selected negative sampling will be used
    as loss function, together with the parameter `negative`. With "hs" hierarchical softmax will be used,
    while with "softmax" (default) the sandard softmax function (the other two are "approximations")

    `negative` or `neg` = if > 0, negative sampling will be used, the int for negative
    specifies how many "noise words" should be drawn (usually between 5-20).
    Default is 5. If set to 0, no negative samping is used. It works only if `loss = ns`

    `workers` or `thread` = use this many worker threads to train the model (=faster training with multicore machines).

    `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
    Only applies when cbow is used.

    `hashfxn` = hash function to use to randomly initialize weights, for increased
    training reproducibility. Default is Python's rudimentary built in hash function.

    `iter` or `epoch` or `max_iter` = number of iterations (epochs) over the corpus. Default is 5.

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

    `pre_trained` can be set with a ``LabeledWord2Vec`` object,
    or a ``Word2Vec`` or ``KeyedVectors'' object (from ``gensim.models``)
    in order to set pre-trained word vectors and vocabularies.
    Use ``partial_fit`` method to learn a supervised model starting from the pre-trained one.

    `bucket` is the maximum number of hashed words, i.e., we limit the feature space to this number,
    ergo we use the hashing trick in the word vocabulary. Default to 0, NO hashing trick

    .. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

    """

    def __init__(self, size=200, alpha=0.05, min_count=5, max_vocab_size=None, sample=1e-3, loss='softmax', negative=5,
                 workers=3, min_alpha=0.0001, cbow_mean=1, hashfxn=hash, null_word=0, trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, iter=5, seed=1, bucket=0, pre_trained=None, **kwargs):
        super(GensimFastText, self).__init__()
        self.set_params(
            size=argument_alternatives(size, kwargs, ('dim',), logger),
            alpha=argument_alternatives(alpha, kwargs, ('lr',), logger),
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            sample=argument_alternatives(sample, kwargs, ('t',), logger),
            workers=argument_alternatives(workers, kwargs, ('thread',), logger),
            min_alpha=min_alpha,
            cbow_mean=cbow_mean,
            hashfxn=hashfxn,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            batch_words=batch_words,
            iter=argument_alternatives(iter, kwargs, ('epoch', 'max_iter'), logger),
            seed=argument_alternatives(seed, kwargs, ('random_state',), logger),
            loss=loss,
            negative=argument_alternatives(negative, kwargs, ('neg',), logger),
            bucket=bucket
        )
        if pre_trained is None:
            params = self.get_params()
            # Convert name conventions from Scikit-learn to Gensim
            del params['pre_trained']
            self._classifier = LabeledWord2Vec(**params)
        else:
            self._classifier = LabeledWord2Vec.load_from(pre_trained)
            self._build_label_info(self._classifier.lvocab.keys())
            self.set_params(
                size=self._classifier.layer1_size,
                alpha=self._classifier.alpha,
                min_count=self._classifier.min_count,
                max_vocab_size=self._classifier.max_vocab_size,
                sample=self._classifier.sample,
                workers=self._classifier.workers,
                min_alpha=self._classifier.min_alpha,
                cbow_mean=self._classifier.cbow_mean,
                hashfxn=self._classifier.hashfxn,
                null_word=self._classifier.null_word,
                sorted_vocab=self._classifier.sorted_vocab,
                batch_words=self._classifier.batch_words,
                iter=self._classifier.iter,
                seed=self._classifier.seed,
                loss='softmax' if self._classifier.softmax else ('hs' if self._classifier.hs else 'ns'),
                negative=self._classifier.negative,
                bucket=self._classifier.bucket,
            )

    @property
    def classifier(self):
        """
        The word embeddings model
        :return: An instance of ``LabeledWord2Vec``, a CBOW model in wich input vectors are words, output vectors are labels.
        """
        return self._classifier

    def fit_embeddings(self, documents):
        """
        Train word embeddings of the classification model, using the same parameter values for classification on Gensim ``Word2Vec``.
        Similar to use a pre-trained model.
        :param documents:
        """
        params = self.get_params()
        del params['pre_trained']
        del params['bucket']
        # Word2Vec has not softmax
        if params['loss'] == 'softmax':
            params['loss'] = 'hs'
        LabeledWord2Vec.init_loss(LabeledWord2Vec(), params, params['loss'])
        del params['loss']
        w2v = Word2Vec(sentences=documents, **params)
        self._classifier = LabeledWord2Vec.load_from(w2v)

    def fit(self, documents, y=None, **fit_params):
        """
        Train the supervised model with a labeled training set
        :param documents: Iterator over lists of words
        :param y: Iterator over lists or single labels, document target values
        :param fit_params: Nothing for now
        :return:
        """
        # TODO if y=None learn a one-class classifier
        self._build_label_info(y, overwrite=True)
        if not self._classifier.wv.vocab:
            self._classifier.build_vocab(documents, self._label_set, trim_rule=self.trim_rule)
        elif not self._classifier.lvocab:
            self._classifier.build_lvocab(self._label_set)
        self._classifier.train(self._data_iter(documents, y))

    def partial_fit(self, documents, y):
        """
        Train more data over the already trained model.
        :param documents: Iterator over lists of words
        :param y: Iterator over lists or single labels, document target values
        """
        if not self._classifier.wv.vocab or not self._classifier.lvocab:
            self.fit(documents, y)
        else:
            self._build_label_info(y)
            size = sum(1 for _ in self._data_iter(documents, y))
            self._classifier.build_vocab(documents, self._label_set, trim_rule=self.trim_rule, update=True)
            self._classifier.train(self._data_iter(documents, y), total_examples=size)

    def _iter_predict(self, documents):
        for doc in documents:
            result = list(score_document_labeled_cbow(self._classifier, document=doc))
            yield result

    def predict_proba(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, a list of label probabilities, which should sum to one for each prediction
        """
        return [self._extract_prediction(prediction) for prediction in self._iter_predict(documents)]

    def decision_function(self, documents):
        return self.predict_proba(documents)

    def predict(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, the one most probable label (i.e. the classification)
        """
        # FIXME it only returns the most probable class, so it is not multi-label (even if the training is)
        return [max(predictions, key=operator.itemgetter(1))[0] for predictions in self._iter_predict(documents)]

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['_classifier'])
        super(GensimFastText, self).save(*args, **kwargs)
        # Get the file name of this object serialization
        if any(args) and 'fname_or_handle' not in kwargs:
            fname = args[0]
            args = args[1:]
        else:
            fname = kwargs['fname_or_handle']
        if not isinstance(fname, basestring):
            fname = fname.name
        fname += CLASSIFIER_FILE_SUFFIX
        kwargs['fname_or_handle'] = fname
        self._classifier.save(*args, **kwargs)

    save.__doc__ = gensim.utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(GensimFastText, cls).load(*args, **kwargs)
        # Get the file name of this object serialization
        if any(args) and 'fname' not in kwargs:
            kwargs['fname'] = args[0]
            args = args[1:]
        kwargs['fname'] += CLASSIFIER_FILE_SUFFIX
        model._classifier = LabeledWord2Vec.load(*args, **kwargs)
        return model


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

    `pretrained_vectors` = pretrained word vectors (.vec file) for supervised learning [None]
    """

    LABEL_PREFIX = '__label__'

    def __init__(self, lr=0.1, lr_update_rate=100, dim=100, ws=5, epoch=5, min_count=1, neg=5, word_ngrams=1,
                 loss='softmax', bucket=0, minn=0, maxn=0, thread=12, t=0.0001, silent=1, encoding='utf-8',
                 pretrained_vectors=None):
        super(FastText, self).__init__()
        self.lr = lr
        self.lr_update_rate = lr_update_rate
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.min_count = min_count
        self.neg = neg
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.bucket = bucket
        self.minn = minn
        self.maxn = maxn
        self.thread = thread
        self.t = t
        self.silent = silent
        self.encoding = encoding
        self.pretrained_vectors = pretrained_vectors or ''
        self._classifier = None
        self._temp_file = None
        self._temp_fname = None

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
                        targets = ' '.join('%s%s' % (self.LABEL_PREFIX, label) for label in y)
                        sample = ' '.join(x)
                        dataset.write(to_unicode(targets + ' ' + sample + '\n'))
            self.fit_file(train_temp.name)

    def fit_file(self, train_path, output_path=None, label_prefix=None):
        """
        Train the model from a training set file
        :param train_path: Txt file of documents, with one or more `LABEL_PREFIX`+`category`
            (default `LABEL_PREFIX` value is "__label__") at the beginning of each line
        followed by the document tokens
        :param output_path: Path where to save the model, a `.bin` extension will be appended
        :param label_prefix:
        """

        def train_classifier(output):
            self._classifier = fasttext.supervised(input_file=train_path, output=output,
                                                   label_prefix=label_prefix or self.LABEL_PREFIX, **self.get_params())

        if output_path is None:
            self._temp_file = tempfile.NamedTemporaryFile(suffix='.bin')
            self._temp_fname = self._temp_file.name
            train_classifier(self._temp_fname[:-4])
        else:
            self._temp_fname = output_path
            train_classifier(output_path)

    def predict_proba(self, documents):
        """
        :param documents: Iterator over lists of words
        :return: For each document, a list of label probabilities, which should sum to one for each prediction
        """
        result = self._classifier.predict_proba(iter(' '.join(d) for d in documents), self._label_count)
        result = [[1. / self._label_count] * self._label_count if not any(r) else self._extract_prediction(r) for r in
                  result]
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

    def __enter__(self):
        return self

    def close(self):
        if hasattr(self, 'temp_file') and self._temp_file is not None:
            self._temp_file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __del__(self):
        return self.close()

    def save(self, *args, **kwargs):
        _temp_file = self._temp_file
        self._temp_file = None
        _classifier = self._classifier
        self._classifier = None
        kwargs['ignore'] = kwargs.get('ignore', ['_classifier', '_temp_file', '_temp_fname'])
        super(FastText, self).save(*args, **kwargs)
        self._temp_file = _temp_file
        self._classifier = _classifier
        if self._temp_fname is not None and self._temp_file:
            # Get the file name of this object serialization
            if any(args) and 'fname_or_handle' not in kwargs:
                fname = args[0]
            else:
                fname = kwargs['fname_or_handle']
            if not isinstance(fname, basestring):
                fname = fname.name
            fname += CLASSIFIER_FILE_SUFFIX + '.bin'
            shutil.copyfile(self._temp_fname, fname)

    save.__doc__ = gensim.utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(FastText, cls).load(*args, **kwargs)
        # Get the file name of this object serialization
        if any(args) and 'fname' not in kwargs:
            kwargs['fname'] = args[0]
        kwargs['fname'] += CLASSIFIER_FILE_SUFFIX + '.bin'
        model._temp_fname = kwargs['fname']
        model._temp_file = None
        model._classifier = fasttext.load_model(kwargs['fname'], label_prefix=cls.LABEL_PREFIX)
        return model

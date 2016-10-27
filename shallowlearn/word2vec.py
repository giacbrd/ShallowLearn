#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Rewritten by Giacomo Berardi <giacbrd.com> Copyright (C) 2016
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

from __future__ import division  # py3 "true division"

import logging
import sys

from gensim import matutils
from gensim.models import Word2Vec
from gensim.models.word2vec import train_cbow_pair, Vocab

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import copy, prod, exp, outer, empty, zeros, ones, uint32, float32 as REAL, dot, sum as np_sum, \
    apply_along_axis
from six.moves import range

__author__ = 'Giacomo Berardi <giacbrd.com>'

logger = logging.getLogger(__name__)

try:

    from .word2vec_inner import train_batch_labeled_cbow, score_document_labeled_cbow as sdlc
    from .word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

    logger.debug('Fast version of {0} is being used'.format(__name__))


    def score_document_labeled_cbow(model, document, label, work=ones(1, dtype=REAL), neu1=None):
        if neu1 is None:
            neu1 = matutils.zeros_aligned(model.layer1_size, dtype=REAL)
        return sdlc(model, document, label, work, neu1)

except ImportError:

    # failed... fall back to plain numpy (20-80x slower training than the above)
    logger.warning('Slow version of {0} is being used'.format(__name__))
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000


    def train_cbow_pair_softmax(model, target, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
        neu1e = zeros(l1.shape)

        target_vect = zeros(model.syn1neg.shape[0])
        target_vect[target.index] = 1.
        l2 = copy(model.syn1neg)
        fa = 1. / (1. + exp(-dot(l1, l2.T)))  # propagate hidden -> output
        ga = (target_vect - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2)  # save error

        if learn_vectors:
            # learn input -> hidden, here for all words in the window separately
            if not model.cbow_mean and input_word_indices:
                neu1e /= len(input_word_indices)
            for i in input_word_indices:
                model.syn0[i] += neu1e * model.syn0_lockf[i]

        return neu1e


    def train_batch_labeled_cbow(model, sentences, alpha, work=None, neu1=None):
        result = 0
        for sentence in sentences:
            document, target = sentence
            word_vocabs = [model.vocab[w] for w in document if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            target_vocabs = [model.lvocab[t] for t in target if t in model.lvocab]
            for target in target_vocabs:
                word2_indices = [w.index for w in word_vocabs]
                l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x vector_size
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                if model.softmax:
                    train_cbow_pair_softmax(model, target, word2_indices, l1, alpha)
                else:
                    train_cbow_pair(model, target, word2_indices, l1, alpha)
            result += len(word_vocabs)
        return result


    def score_document_labeled_cbow(model, document, label, work=None, neu1=None):

        word_vocabs = [model.vocab[w] for w in document if w in model.vocab]

        if label in model.lvocab:
            target = model.lvocab[label]
        else:
            raise KeyError('Class label "%s" not found in the vocabulary' % label)

        word2_indices = [word2.index for word2 in word_vocabs]
        l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x layer1_size
        if word2_indices and model.cbow_mean:
            l1 /= len(word2_indices)
        return score_cbow_labeled_pair(model, target, word2_indices, l1)


    def score_cbow_labeled_pair(model, target, word2_indices, l1):
        if model.hs:
            l2a = model.syn1[target.point]  # 2d matrix, codelen x layer1_size
            sgn = (-1.0) ** target.code  # ch function, 0-> 1, 1 -> -1
            prob = 1.0 / (1.0 + exp(-sgn * dot(l1, l2a.T)))
        # Softmax
        else:
            def exp_dot(x):
                return exp(dot(l1, x.T))

            prob_num = exp_dot(model.syn1neg[target.index])
            prob_den = np_sum(apply_along_axis(exp_dot, 1, model.syn1neg))
            prob = prob_num / prob_den
        return prod(prob)


class LabeledWord2Vec(Word2Vec):
    def __init__(self, loss='softmax', **kwargs):
        """
        Exactly as the parent class `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_.
        Some parameter values are overwritten (e.g. sg=0 because we never use skip-gram here), look at the code for details.

        `loss` = one value in {ns, hs, softmax}. If "ns" is selected negative sampling will be used
        as loss function, together with the parameter `negative`. With "hs" hierarchical softmax will be used,
        while with "softmax" (default) the sandard softmax function (the other two are "approximations").
         The `hs` argument does not exist anymore.

        It basically builds two vocabularies, one for the sample words and one for the labels,
        so that the input layer is only made of words, while the output layer is only made of labels.
        **Parent class methods that are not overridden here are not tested and not safe to use**.
        """
        self.lvocab = {}  # Vocabulary of labels only
        self.index2label = []
        kwargs['sg'] = 0
        kwargs['window'] = sys.maxsize
        kwargs['sentences'] = None
        self.softmax = False
        if loss == 'hs':
            kwargs['hs'] = 1
            kwargs['negative'] = 0
        elif loss == 'ns':
            kwargs['hs'] = 0
            assert kwargs['negative'] > 0
        elif loss == 'softmax':
            kwargs['hs'] = 0
            kwargs['negative'] = 0
            self.softmax = True
        else:
            raise ValueError('loss argument must be set with "ns", "hs" or "softmax"')
        super(LabeledWord2Vec, self).__init__(**kwargs)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence[0]) for sentence in job)

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences (documents and labels in LabeledWord2Vec).
        Return 2-tuple `(effective word count after ignoring unknown words and sentence length trimming,
        total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            raise NotImplementedError('Supervised learning in fastText is based only on the CBOW model')
        else:
            tally += train_batch_labeled_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)

    # TODO use TaggedDocument from Gensim?
    # FIXME pass just an iterator over (doc, label) like for train
    def build_vocab(self, sentences, labels, keep_raw_vocab=False, trim_rule=None, progress_per=10000):
        """
        Build vocabularies from a sequence of sentences/documents (can be a once-only generator stream) and the set of labels.
        Each sentence must be a list of unicode strings. `labels` is an iterable over the label names.

        """
        class FakeSelf(LabeledWord2Vec):
            def __init__(self, max_vocab_size, min_count, sample, estimate_memory):
                self.max_vocab_size = max_vocab_size
                self.corpus_count = 0
                self.raw_vocab = None
                self.vocab = {}
                self.min_count = min_count
                self.index2word = []
                self.sample = sample
                self.estimate_memory = estimate_memory

        # Build words and labels vocabularies in two different oobjects
        # FIXME set the right estimate memory for labels
        labels_vocab = FakeSelf(sys.maxsize, 0, 0, self.estimate_memory)
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.__class__.scan_vocab(labels_vocab, [labels], progress_per=progress_per, trim_rule=None)
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)
        self.__class__.scale_vocab(labels_vocab, min_count=None, sample=None, keep_raw_vocab=False, trim_rule=None)
        self.lvocab = labels_vocab.vocab
        self.index2label = labels_vocab.index2word
        # If we want to sample more negative labels that their count
        if self.negative and self.negative >= len(self.index2label):
            self.negative = len(self.index2label) - 1
        self.finalize_vocab()  # build tables & arrays

    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""

        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab:
            self.sort_vocab()
        if self.hs:
            class FakeSelf(LabeledWord2Vec):
                def __init__(self, vocab):
                    self.vocab = vocab

            # add info about each word's Huffman encoding
            self.__class__.create_binary_tree(FakeSelf(self.lvocab))
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()

        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial input/projection and hidden weights
        self.reset_weights()

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in range(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        # Output layer is only made of labels
        if self.hs:
            self.syn1 = zeros((len(self.lvocab), self.layer1_size), dtype=REAL)
        # Use syn1neg also for softmax outputs
        if self.negative or self.softmax:
            self.syn1neg = zeros((len(self.lvocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None
        self.syn0_lockf = ones(len(self.vocab), dtype=REAL)  # zeros suppress learning

    def reset_from(self, other_model):
        """
        Borrow shareable pre-built structures (like vocab) from the other_model. Useful
        if testing multiple models in parallel on the same corpus.
        """
        self.lvocab = getattr(other_model, 'lvocab', other_model.vocab)
        self.index2label = getattr(other_model, 'index2label', other_model.index2word)
        super(LabeledWord2Vec, self).reset_from(other_model)

    def make_cum_table(self, power=0.75, domain=2 ** 31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary label counts for
        drawing random labels in the negative-sampling training routines.

        To draw a label index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.index2label)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.lvocab[word].count ** power for word in self.lvocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.lvocab[self.index2label[word_index]].count ** power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.
        return super(LabeledWord2Vec, self).train(sentences, total_words, word_count,
                                                  total_examples, queue_factor, report_delay)

    def score(self, **kwargs):
        raise NotImplementedError('This method has no reason to exist in this class (for now)')

    def save_word2vec_format(self, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    @classmethod
    def load_word2vec_format(cls, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    def intersect_word2vec_format(self, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    def accuracy(self, **kwargs):
        raise NotImplementedError('This method has no reason to exist in this class (for now)')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import io
import pickle

import numpy
import pytest
from six.moves import zip

from shallowlearn.models import BaseClassifier
from shallowlearn.word2vec import LabeledWord2Vec, score_document_labeled_cbow
from .resources import dataset_samples
from .resources import dataset_targets

__author__ = 'Giacomo Berardi <giacbrd.com>'


@pytest.fixture
def small_model():
    model = LabeledWord2Vec(iter=1, size=30, min_count=0)
    model.build_vocab(dataset_samples, frozenset(
        target for targets in dataset_targets for target in BaseClassifier._target_list(targets)))
    model.train(zip(dataset_samples, dataset_targets))
    return model


def test_init():
    model1 = LabeledWord2Vec()
    model2 = LabeledWord2Vec(iter=1, size=50)
    model3 = LabeledWord2Vec(seed=66)
    assert model1 != model2
    assert model1 != model3


def test_vocabulary(small_model):
    assert 'to' in small_model.vocab
    assert frozenset(('a', 'b', 'c')) == frozenset(small_model.lvocab.keys())
    assert max(v.index for v in small_model.lvocab.values()) == 2


def test_matrices(small_model):
    assert small_model.syn1.shape[0] == 3
    assert small_model.syn1.shape[1] == 30
    assert not hasattr(small_model, 'syn1neg')


def test_serializzation(small_model):
    with io.BytesIO() as fileobj:
        pickle.dump(small_model, fileobj)
        fileobj.seek(0)
        loaded = pickle.load(fileobj)
        assert all(str(loaded.vocab[w]) == str(small_model.vocab[w]) for w in small_model.vocab)
        assert all(str(loaded.lvocab[w]) == str(small_model.lvocab[w]) for w in small_model.lvocab)
        assert numpy.array_equiv(loaded.syn1, small_model.syn1)
        assert numpy.array_equiv(loaded.syn0, small_model.syn0)


def test_learning_functions(small_model):
    a = score_document_labeled_cbow(small_model, ('study', 'to', 'learn'), 'a')
    b = score_document_labeled_cbow(small_model, ('study', 'to', 'learn'), 'b')
    c = score_document_labeled_cbow(small_model, ('study', 'to', 'learn'), 'c')
    assert round(a + b + c) == 1.

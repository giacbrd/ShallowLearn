#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import pytest
from shallowlearn.models import GensimFastText, FastText
from tests.resources import dataset_targets, dataset_samples
from tests.test_word2vec import bunch_of_models

__author__ = 'Giacomo Berardi <giacbrd.com>'


@pytest.fixture
def bunch_of_gensim_classifiers():
    return [GensimFastText(pre_trained=m) for m in bunch_of_models()]


@pytest.fixture
def bunch_of_fasttext_classifiers():
    models = []
    for kwarg in ({'loss': 'softmax'},):
        models.extend([
            FastText(epoch=1, dim=30, min_count=0, **kwarg),
            FastText(epoch=1, lr=1.0, dim=300, min_count=0, **kwarg),
            FastText(epoch=1, dim=100, min_count=1, **kwarg),
            FastText(epoch=1, dim=100, min_count=0, t=0, **kwarg),
            FastText(epoch=3, dim=100, min_count=0, **kwarg),
            FastText(epoch=5, thread=1, dim=100, min_count=0, **kwarg)
        ])
    for model in models:
        model.fit(dataset_samples, dataset_targets)
    return models


def _predict(model):
    example = [('study', 'to', 'learn', 'me', 'study', 'to', 'learn', 'me', 'machine', 'learning')]
    pr = model.predict_proba(example)
    assert pr[0][0][1] > .33
    p = model.predict(example)
    if pr[0][0][1] - pr[0][1][1] > .000001:
        assert p == ['aa'] or p == ['b']

def test_gensim_predict(bunch_of_gensim_classifiers):
    for model in bunch_of_gensim_classifiers:
        _predict(model)


def test_gensim_fit(bunch_of_gensim_classifiers):
    for model in bunch_of_gensim_classifiers:
        params = model.get_params()
        params['pre_trained'] = None
        clf = GensimFastText(**params)
        clf.fit(dataset_samples, dataset_targets)
        _predict(model)


def test_fasttext_predict(bunch_of_fasttext_classifiers):
    for model in bunch_of_fasttext_classifiers:
        _predict(model)

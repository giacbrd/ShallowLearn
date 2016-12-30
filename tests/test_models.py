#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import io
import tempfile

import pytest

from shallowlearn.models import GensimFastText, FastText
from tests.resources import dataset_targets, dataset_samples, pre_docs
from tests.test_word2vec import bunch_of_models

__author__ = 'Giacomo Berardi <giacbrd.com>'


@pytest.fixture
def bunch_of_gensim_classifiers():
    return [GensimFastText(pre_trained=m) for m in bunch_of_models()]


@pytest.fixture
def bunch_of_fasttext_classifiers():
    models = []
    for kwarg in ({'loss': 'softmax'}, {'loss': 'softmax', 'bucket': 5}, {'loss': 'softmax', 'bucket': 100}):
        models.extend([
            FastText(epoch=1, dim=30, min_count=0, **kwarg),
            FastText(epoch=1, lr=1.0, dim=100, min_count=0, **kwarg),
            FastText(epoch=1, dim=50, min_count=1, **kwarg),
            FastText(epoch=1, dim=50, min_count=0, t=0, **kwarg),
            FastText(epoch=3, dim=10, min_count=0, **kwarg),
            FastText(epoch=5, thread=1, dim=50, min_count=0, **kwarg)
        ])
    for model in models:
        model.fit(dataset_samples, dataset_targets)
    return models


def _predict(model):
    example = [('study', 'to', 'learn', 'me', 'study', 'to', 'learn', 'me', 'machine', 'learning')]
    pr = model.predict_proba(example)
    pr = list(pr[0])
    pr.sort(reverse=True)
    assert pr[0] > .33
    p = model.predict(example)
    if pr[0] - pr[1] > .01:
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
        _predict(clf)


def test_fasttext_predict(bunch_of_fasttext_classifiers):
    for model in bunch_of_fasttext_classifiers:
        _predict(model)


def _persistence(classifiers):
    example = [('supervised', 'faster', 'is', 'machine', 'important')]
    for model in classifiers:
        with tempfile.NamedTemporaryFile() as temp:
            model.save(temp.name)
            loaded = model.load(temp.name)
            assert model.get_params() == loaded.get_params()
            assert model.predict(example) == loaded.predict(example)
        with tempfile.NamedTemporaryFile() as temp:
            with io.open(temp.name, 'wb') as temp_out:
                model.save(temp_out)
            loaded = model.load(temp.name)
            assert model.get_params() == loaded.get_params()
            assert model.predict(example) == loaded.predict(example)


def test_persistence_gensim(bunch_of_gensim_classifiers):
    return _persistence(bunch_of_gensim_classifiers)


def test_persistence_fasttext(bunch_of_fasttext_classifiers):
    return _persistence(bunch_of_fasttext_classifiers)


def test_duplicate_arguments():
    clf = GensimFastText(seed=9, random_state=10, neg=20, epoch=25, negative=11)
    params = clf.get_params()
    assert params['seed'] == clf.seed == 10
    assert params['negative'] == clf.negative == 20
    assert params['iter'] == clf.iter == 25


def test_fit_embeddings(bunch_of_gensim_classifiers):
    for model in bunch_of_gensim_classifiers:
        model.fit_embeddings(pre_docs)
        model.fit(dataset_samples, dataset_targets)
        _predict(model)

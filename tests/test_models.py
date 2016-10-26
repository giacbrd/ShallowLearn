#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import pytest
from shallowlearn.models import GensimFastText
from tests.resources import dataset_targets, dataset_samples
from tests.test_word2vec import bunch_of_models

__author__ = 'Giacomo Berardi <giacbrd.com>'


@pytest.fixture
def bunch_of_classifiers():
    return [GensimFastText(pre_trained=m) for m in bunch_of_models()]


def _predict(model):
    example = [('study', 'to', 'learn', 'me', 'study', 'to', 'learn', 'me', 'machine', 'learning')]
    p = model.predict(example)
    assert p == ['a'] or p == ['b']
    p = model.predict_proba(example)
    assert p[0][0][0] == 'a' or p[0][0][0] == 'b'
    assert p[0][0][1] > .3333333


def test_predict(bunch_of_classifiers):
    for model in bunch_of_classifiers:
        _predict(model)


def test_fit(bunch_of_classifiers):
    for model in bunch_of_classifiers:
        params = model.get_params()
        params['pre_trained'] = None
        clf = GensimFastText(**params)
        clf.fit(dataset_samples, dataset_targets)
        _predict(model)
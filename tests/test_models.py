#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import pytest
from shallowlearn.models import GensimFastText
from tests.test_word2vec import bunch_of_models

__author__ = 'Giacomo Berardi <giacbrd.com>'


@pytest.fixture
def bunch_of_classifiers():
    return [GensimFastText(pre_trained=m) for m in bunch_of_models()]


def test_predict(bunch_of_classifiers):
    for model in bunch_of_classifiers:
        p = model.predict([('study', 'to', 'learn')])
        assert p == ['a']
        p = model.predict_proba([('study', 'to', 'learn')])
        assert p[0][0][0] == 'a'
        assert p[0][0][1] >= .34
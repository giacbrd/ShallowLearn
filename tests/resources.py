#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

__author__ = 'Giacomo Berardi <giacbrd.com>'

pre_docs = (
    ('i', 'study', 'machine', 'learning', 'to', 'drive', 'a', 'car'),
    ('important', 'topics', 'of', 'this', 'week', 'are', 'today', 'cheers'),
    ('like', 'my', 'faster', 'bike', 'supervised', 'is', 'dedicated', 'to', 'me'),
)
dataset_samples = (
    ('i', 'like', 'to', 'study', 'machine', 'learning'),
    ('machine', 'learning', 'is', 'important'),
    ('my', 'car', 'is', 'faster', 'than', 'your', 'bike'),
    ('supervised', 'machine', 'learning', 'is', 'the', 'topic', 'of', 'today'),
    ('my', 'study', 'is', 'dedicated', 'to', 'me')
)
dataset_targets = (('aa', 'b'), 'b', 'cc', 'b', 'aa')
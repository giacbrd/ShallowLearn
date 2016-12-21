#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html
import numpy


def argument_alternatives(original_value, kwargs, alternative_names, logger):
    final_value = original_value
    for name in reversed(alternative_names):
        if name in kwargs and kwargs[name] != original_value:
            logger.warning('%s parameter overwrites the already set value of %s', name, final_value)
            final_value = kwargs[name]
    return final_value


class HashIter(object):
    def __init__(self, documents, bucket):
        self.bucket = bucket
        self.documents = documents

    def __iter__(self):
        for doc in self.documents:
            yield [self.ft_hash(word) % self.bucket for word in doc]

    @classmethod
    def ft_hash(cls, string):
        # Reproduces hash method used in fastText
        h = numpy.uint32(2166136261)
        for c in string:
            h ^= numpy.uint32(ord(c))
            h *= numpy.uint32(16777619)
        return h

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html

import numpy

try:
    basestring = basestring
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str, bytes)


def argument_alternatives(original_value, kwargs, alternative_names, logger):
    final_value = original_value
    for name in reversed(alternative_names):
        if name in kwargs and kwargs[name] != original_value:
            logger.warning('%s parameter overwrites the already set value of %s', name, final_value)
            final_value = kwargs[name]
    return final_value


class HashIter(object):
    def __init__(self, documents, bucket, with_labels=False):
        self.with_labels = with_labels
        self.bucket = bucket
        self.documents = documents

    def __iter__(self):
        for doc in self.documents:
            if self.with_labels:
                yield (self.hash_doc(doc[0], self.bucket), doc[1])
            else:
                yield self.hash_doc(doc, self.bucket)

    @classmethod
    def hash_doc(cls, document, bucket):
        return [cls.hash(word) % bucket for word in document]

    @classmethod
    def hash(cls, word):
        # Reproduces hash method used in fastText
        h = numpy.uint32(2166136261)
        for c in bytearray(word.encode('utf-8', errors='ignore')):
            h ^= numpy.uint32(c)
            h *= numpy.uint32(16777619)
        return h

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Giacomo Berardi <giacbrd.com>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/lgpl.html


def argument_alternatives(original_value, kwargs, alternative_names, logger):
    final_value = original_value
    for name in reversed(alternative_names):
        if name in kwargs and kwargs[name] != original_value:
            logger.warn('{} parameter overwrites the already set value of {}'.format(name, final_value))
            final_value = kwargs[name]
    return final_value

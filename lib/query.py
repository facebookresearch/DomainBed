# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Small query library."""

import collections
import inspect
import json
import types
import unittest
import warnings
import math

import numpy as np


def make_selector_fn(selector):
    """
    If selector is a function, return selector.
    Otherwise, return a function corresponding to the selector string. Examples
    of valid selector strings and the corresponding functions:
        x       lambda obj: obj['x']
        x.y     lambda obj: obj['x']['y']
        x,y     lambda obj: (obj['x'], obj['y'])
    """
    if isinstance(selector, str):
        if ',' in selector:
            parts = selector.split(',')
            part_selectors = [make_selector_fn(part) for part in parts]
            return lambda obj: tuple(sel(obj) for sel in part_selectors)
        elif '.' in selector:
            parts = selector.split('.')
            part_selectors = [make_selector_fn(part) for part in parts]
            def f(obj):
                for sel in part_selectors:
                    obj = sel(obj)
                return obj
            return f
        else:
            key = selector.strip()
            return lambda obj: obj[key]
    elif isinstance(selector, types.FunctionType):
        return selector
    else:
        raise TypeError

def hashable(obj):
    try:
        hash(obj)
        return obj
    except TypeError:
        return json.dumps({'_':obj}, sort_keys=True)

class Q(object):
    def __init__(self, list_):
        super(Q, self).__init__()
        self._list = list_

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        return self._list[key]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._list == other._list
        else:
            return self._list == other

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return repr(self._list)

    def _append(self, item):
        """Unsafe, be careful you know what you're doing."""
        self._list.append(item)

    def group(self, selector):
        """
        Group elements by selector and return a list of (group, group_records)
        tuples.
        """
        selector = make_selector_fn(selector)
        groups = {}
        for x in self._list:
            group = selector(x)
            group_key = hashable(group)
            if group_key not in groups:
                groups[group_key] = (group, Q([]))
            groups[group_key][1]._append(x)
        results = [groups[key] for key in sorted(groups.keys())]
        return Q(results)

    def group_map(self, selector, fn):
        """
        Group elements by selector, apply fn to each group, and return a list
        of the results.
        """
        return self.group(selector).map(fn)

    def map(self, fn):
        """
        map self onto fn. If fn takes multiple args, tuple-unpacking
        is applied.
        """
        if len(inspect.signature(fn).parameters) > 1:
            return Q([fn(*x) for x in self._list])
        else:
            return Q([fn(x) for x in self._list])

    def select(self, selector):
        selector = make_selector_fn(selector)
        return Q([selector(x) for x in self._list])

    def min(self):
        return min(self._list)

    def max(self):
        return max(self._list)

    def sum(self):
        return sum(self._list)

    def len(self):
        return len(self._list)

    def mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(self._list))

    def std(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.std(self._list))

    def mean_std(self):
        return (self.mean(), self.std())

    def argmax(self, selector):
        selector = make_selector_fn(selector)
        return max(self._list, key=selector)

    def filter(self, fn):
        return Q([x for x in self._list if fn(x)])

    def filter_equals(self, selector, value):
        """like [x for x in y if x.selector == value]"""
        selector = make_selector_fn(selector)
        return self.filter(lambda r: selector(r) == value)

    def filter_not_none(self):
        return self.filter(lambda r: r is not None)

    def filter_not_nan(self):
        return self.filter(lambda r: not np.isnan(r))

    def flatten(self):
        return Q([y for x in self._list for y in x])

    def unique(self):
        result = []
        result_set = set()
        for x in self._list:
            hashable_x = hashable(x)
            if hashable_x not in result_set:
                result_set.add(hashable_x)
                result.append(x)
        return Q(result)

    def sorted(self, key=None):
        if key is None:
            key = lambda x: x
        def key2(x):
            x = key(x)
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                return float('-inf')
            else:
                return x
        return Q(sorted(self._list, key=key2))

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from domainbed.lib.query import Q, make_selector_fn

class TestQuery(unittest.TestCase):
    def test_everything(self):
        numbers = Q([1, 4, 2])
        people = Q([
            {'name': 'Bob', 'age': 40},
            {'name': 'Alice', 'age': 20},
            {'name': 'Bob', 'age': 10}
        ])

        self.assertEqual(numbers.select(lambda x: 2*x), [2, 8, 4])

        self.assertEqual(numbers.min(), 1)
        self.assertEqual(numbers.max(), 4)
        self.assertEqual(numbers.mean(), 7/3)

        self.assertEqual(people.select('name'), ['Bob', 'Alice', 'Bob'])

        self.assertEqual(
            set(people.group('name').map(lambda _,g: g.select('age').mean())),
            set([25, 20])
        )

        self.assertEqual(people.argmax('age'), people[0])

    def test_group_by_unhashable(self):
        jobs = Q([
            {'hparams': {1:2}, 'score': 3},
            {'hparams': {1:2}, 'score': 4},
            {'hparams': {2:4}, 'score': 5}
        ])
        grouped = jobs.group('hparams')
        self.assertEqual(grouped, [
            ({1:2}, [jobs[0], jobs[1]]),
            ({2:4}, [jobs[2]])
        ])

    def test_comma_selector(self):
        struct = {'a': {'b': 1}, 'c': 2}
        fn = make_selector_fn('a.b,c')
        self.assertEqual(fn(struct), (1, 2))

    def test_unique(self):
        numbers = Q([1,2,1,3,2,1,3,1,2,3])
        self.assertEqual(numbers.unique(), [1,2,3])

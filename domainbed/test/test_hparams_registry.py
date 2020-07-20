# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import itertools

from domainbed import hparams_registry
from domainbed import datasets
from domainbed import algorithms

from parameterized import parameterized

class TestHparamsRegistry(unittest.TestCase):

    @parameterized.expand(itertools.product(algorithms.ALGORITHMS, datasets.DATASETS))
    def test_random_hparams_deterministic(self, algorithm_name, dataset_name):
        """Test that hparams_registry.random_hparams is deterministic"""
        a = hparams_registry.random_hparams(algorithm_name, dataset_name, 0)
        b = hparams_registry.random_hparams(algorithm_name, dataset_name, 0)
        self.assertEqual(a.keys(), b.keys())
        for key in a.keys():
            self.assertEqual(a[key], b[key], key)

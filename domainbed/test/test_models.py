# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import unittest
import uuid

import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import networks
from domainbed.test import helpers

from parameterized import parameterized


class TestAlgorithms(unittest.TestCase):

    @parameterized.expand(itertools.product(helpers.DEBUG_DATASETS, algorithms.ALGORITHMS))
    def test_init_update_predict(self, dataset_name, algorithm_name):
        """Test that a given algorithm inits, updates and predicts without raising
        errors."""
        batch_size = 8
        hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)('', [], hparams)
        minibatches = helpers.make_minibatches(dataset, batch_size)
        algorithm_class = algorithms.get_algorithm_class(algorithm_name)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset),
            hparams).cuda()
        for _ in range(3):
            self.assertIsNotNone(algorithm.update(minibatches))
        algorithm.eval()
        self.assertEqual(list(algorithm.predict(minibatches[0][0]).shape),
            [batch_size, dataset.num_classes])

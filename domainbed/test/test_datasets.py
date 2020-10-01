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

from parameterized import parameterized

from domainbed.test import helpers

class TestDatasets(unittest.TestCase):

    @parameterized.expand(itertools.product(datasets.DATASETS))
    @unittest.skipIf('DATA_DIR' not in os.environ, 'needs DATA_DIR environment '
        'variable')
    def test_dataset_erm(self, dataset_name):
        """
        Test that ERM can complete one step on a given dataset without raising
        an error.
        Also test that num_environments() works correctly.
        """
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)(
            os.environ['DATA_DIR'], [], hparams)
        self.assertEqual(datasets.num_environments(dataset_name),
            len(dataset))
        algorithm = algorithms.get_algorithm_class('ERM')(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset),
            hparams).cuda()
        minibatches = helpers.make_minibatches(dataset, batch_size)
        algorithm.update(minibatches)

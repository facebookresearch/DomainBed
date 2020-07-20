# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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

import datasets
import hparams_registry
import algorithms
import networks
from parameterized import parameterized

import test.helpers

class TestNetworks(unittest.TestCase):

    @parameterized.expand(itertools.product(test.helpers.DEBUG_DATASETS))
    def test_featurizer(self, dataset_name):
        """Test that Featurizer() returns a module which can take a
        correctly-sized input and return a correctly-sized output."""
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)('', [], hparams)
        input_ = test.helpers.make_minibatches(dataset, batch_size)[0][0]
        input_shape = dataset.input_shape
        algorithm = networks.Featurizer(input_shape, hparams).cuda()
        output = algorithm(input_)
        self.assertEqual(list(output.shape), [batch_size, algorithm.n_outputs])

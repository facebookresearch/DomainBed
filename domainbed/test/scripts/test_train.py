# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# import argparse
# import itertools
import json
import os
import subprocess
# import sys
# import time
import unittest
import uuid

import torch

# import datasets
# import hparams_registry
# import algorithms
# import networks
# from parameterized import parameterized

# import test.helpers

class TestTrain(unittest.TestCase):

    @unittest.skipIf('DATA_DIR' not in os.environ, 'needs DATA_DIR environment '
        'variable')
    def test_end_to_end(self):
        """Test that train.py successfully completes one step"""
        output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)

        subprocess.run(f'python -m domainbed.scripts.train --dataset RotatedMNIST '
            f'--data_dir={os.environ["DATA_DIR"]} --output_dir={output_dir} '
            f'--steps=501', shell=True)

        with open(os.path.join(output_dir, 'results.jsonl')) as f:
            lines = [l[:-1] for l in f]
            last_epoch = json.loads(lines[-1])
            self.assertEqual(last_epoch['step'], 500)
            # Conservative values; anything lower and something's likely wrong.
            self.assertGreater(last_epoch['env0_in_acc'], 0.80)
            self.assertGreater(last_epoch['env1_in_acc'], 0.95)
            self.assertGreater(last_epoch['env2_in_acc'], 0.95)
            self.assertGreater(last_epoch['env3_in_acc'], 0.95)
            self.assertGreater(last_epoch['env3_in_acc'], 0.95)

        with open(os.path.join(output_dir, 'out.txt')) as f:
            text = f.read()
            self.assertTrue('500' in text)

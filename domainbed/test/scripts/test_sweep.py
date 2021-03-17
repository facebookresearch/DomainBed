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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import networks
from domainbed.test import helpers
from domainbed.scripts import sweep

from parameterized import parameterized

class TestSweep(unittest.TestCase):

    def test_job(self):
        """Test that a newly-created job has valid
        output_dir, state, and command_str properties."""
        train_args = {'foo': 'bar'}
        sweep_output_dir = f'/tmp/{str(uuid.uuid4())}'
        job = sweep.Job(train_args, sweep_output_dir)
        self.assertTrue(job.output_dir.startswith(sweep_output_dir))
        self.assertEqual(job.state, sweep.Job.NOT_LAUNCHED)
        self.assertEqual(job.command_str,
            f'python -m domainbed.scripts.train --foo bar --output_dir {job.output_dir}')

    def test_job_launch(self):
        """Test that launching a job calls the launcher_fn with appropariate
        arguments, and sets the job to INCOMPLETE state."""
        train_args = {'foo': 'bar'}
        sweep_output_dir = f'/tmp/{str(uuid.uuid4())}'
        job = sweep.Job(train_args, sweep_output_dir)

        launcher_fn_called = False
        def launcher_fn(commands):
            nonlocal launcher_fn_called
            launcher_fn_called = True
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0], job.command_str)

        sweep.Job.launch([job], launcher_fn)
        self.assertTrue(launcher_fn_called)

        job = sweep.Job(train_args, sweep_output_dir)
        self.assertEqual(job.state, sweep.Job.INCOMPLETE)

    def test_job_delete(self):
        """Test that deleting a launched job returns it to the NOT_LAUNCHED
        state"""
        train_args = {'foo': 'bar'}
        sweep_output_dir = f'/tmp/{str(uuid.uuid4())}'
        job = sweep.Job(train_args, sweep_output_dir)
        sweep.Job.launch([job], (lambda commands: None))
        sweep.Job.delete([job])

        job = sweep.Job(train_args, sweep_output_dir)
        self.assertEqual(job.state, sweep.Job.NOT_LAUNCHED)


    def test_make_args_list(self):
        """Test that, for a typical input, make_job_list returns a list
        of the correct length"""
        args_list = sweep.make_args_list(
            n_trials=2,
            dataset_names=['Debug28'],
            algorithms=['ERM'],
            n_hparams_from=0,
            n_hparams=3,
            steps=123,
            data_dir='/tmp/data',
            task='domain_generalization',
            holdout_fraction=0.2,
            single_test_envs=False,
            hparams=None
        )
        assert(len(args_list) == 2*3*(3+3))

    @unittest.skipIf('DATA_DIR' not in os.environ, 'needs DATA_DIR environment '
        'variable')
    def test_end_to_end(self):
        output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        result = subprocess.run(f'python -m domainbed.scripts.sweep launch '
            f'--data_dir={os.environ["DATA_DIR"]} --output_dir={output_dir} '
            f'--algorithms ERM --datasets Debug28 --n_hparams 1 --n_trials 1 '
            f'--command_launcher dummy --skip_confirmation',
            shell=True, capture_output=True)
        stdout_lines = result.stdout.decode('utf8').split("\n")
        dummy_launcher_lines = [l for l in stdout_lines
            if l.startswith('Dummy launcher:')]
        self.assertEqual(len(dummy_launcher_lines), 6)

        # Now run it again and make sure it doesn't try to relaunch those jobs
        result = subprocess.run(f'python -m domainbed.scripts.sweep launch '
            f'--data_dir={os.environ["DATA_DIR"]} --output_dir={output_dir} '
            f'--algorithms ERM --datasets Debug28 --n_hparams 1 --n_trials 1 '
            f'--command_launcher dummy --skip_confirmation',
            shell=True, capture_output=True)
        stdout_lines = result.stdout.decode('utf8').split("\n")
        dummy_launcher_lines = [l for l in stdout_lines
            if l.startswith('Dummy launcher:')]
        self.assertEqual(len(dummy_launcher_lines), 0)

        # Delete the incomplete jobs, try launching again, and make sure they
        # get relaunched.
        subprocess.run(f'python -m domainbed.scripts.sweep delete_incomplete '
            f'--data_dir={os.environ["DATA_DIR"]} --output_dir={output_dir} '
            f'--algorithms ERM --datasets Debug28 --n_hparams 1 --n_trials 1 '
            f'--command_launcher dummy --skip_confirmation',
            shell=True, capture_output=True)

        result = subprocess.run(f'python -m domainbed.scripts.sweep launch '
            f'--data_dir={os.environ["DATA_DIR"]} --output_dir={output_dir} '
            f'--algorithms ERM --datasets Debug28 --n_hparams 1 --n_trials 1 '
            f'--command_launcher dummy --skip_confirmation',
            shell=True, capture_output=True)
        stdout_lines = result.stdout.decode('utf8').split("\n")
        dummy_launcher_lines = [l for l in stdout_lines
            if l.startswith('Dummy launcher:')]
        self.assertEqual(len(dummy_launcher_lines), 6)

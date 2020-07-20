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

from domainbed import model_selection
from domainbed.lib.query import Q

from parameterized import parameterized

def make_record(step, hparams_seed, envs):
    """envs is a list of (in_acc, out_acc, is_test_env) tuples"""
    result = {
        'args': {'test_envs': [], 'hparams_seed': hparams_seed},
        'step': step
    }
    for i, (in_acc, out_acc, is_test_env) in enumerate(envs):
        if is_test_env:
            result['args']['test_envs'].append(i)
        result[f'env{i}_in_acc'] = in_acc
        result[f'env{i}_out_acc'] = out_acc
    return result

class TestSelectionMethod(unittest.TestCase):

    class MySelectionMethod(model_selection.SelectionMethod):
        @classmethod
        def run_acc(self, run_records):
            return {
                'val_acc': run_records[0]['env0_out_acc'],
                'test_acc': run_records[0]['env0_in_acc']
            }

    def test_sweep_acc(self):
        sweep_records = Q([
            make_record(0, 0, [(0.7, 0.8, True)]),
            make_record(0, 1, [(0.9, 0.5, True)])
        ])

        self.assertEqual(
            self.MySelectionMethod.sweep_acc(sweep_records),
            0.7
        )

    def test_sweep_acc_empty(self):
        self.assertEqual(
            self.MySelectionMethod.sweep_acc(Q([])),
            None
        )

class TestOracleSelectionMethod(unittest.TestCase):

    def test_run_acc_best_first(self):
        """Test run_acc() when the run has two records and the best one comes
        first"""
        run_records = Q([
            make_record(0, 0, [(0.75, 0.70, True)]),
            make_record(1, 0, [(0.65, 0.60, True)])
        ])
        self.assertEqual(
            model_selection.OracleSelectionMethod.run_acc(run_records),
            {'val_acc': 0.60, 'test_acc': 0.65}
        )

    def test_run_acc_best_last(self):
        """Test run_acc() when the run has two records and the best one comes
        last"""
        run_records = Q([
            make_record(0, 0, [(0.75, 0.70, True)]),
            make_record(1, 0, [(0.85, 0.80, True)])
        ])
        self.assertEqual(
            model_selection.OracleSelectionMethod.run_acc(run_records),
            {'val_acc': 0.80, 'test_acc': 0.85}
        )

    def test_run_acc_empty(self):
        """Test run_acc() when there are no valid records to choose from."""
        self.assertEqual(
            model_selection.OracleSelectionMethod.run_acc(Q([])),
            None
        )

class TestIIDAccuracySelectionMethod(unittest.TestCase):

    def test_run_acc(self):
        run_records = Q([
            make_record(0, 0,
                [(0.1, 0.2, True), (0.5, 0.6, False), (0.6, 0.7, False)]),
            make_record(1, 0,
                [(0.3, 0.4, True), (0.6, 0.7, False), (0.7, 0.8, False)]),
        ])
        self.assertEqual(
            model_selection.IIDAccuracySelectionMethod.run_acc(run_records),
            {'val_acc': 0.75, 'test_acc': 0.3}
        )

    def test_run_acc_empty(self):
        self.assertEqual(
            model_selection.IIDAccuracySelectionMethod.run_acc(Q([])),
            None)

class TestLeaveOneOutSelectionMethod(unittest.TestCase):

    def test_run_acc(self):
        run_records = Q([
            make_record(0, 0,
                [(0.1, 0., True), (0.0, 0., False), (0.0, 0., False)]),
            make_record(0, 0,
                [(0.0, 0., True), (0.5, 0., True), (0., 0., False)]),
            make_record(0, 0,
                [(0.0, 0., True), (0.0, 0., False), (0.6, 0., True)]),
        ])
        self.assertEqual(
            model_selection.LeaveOneOutSelectionMethod.run_acc(run_records),
            {'val_acc': 0.55, 'test_acc': 0.1}
        )

    def test_run_acc_empty(self):
        run_records = Q([
            make_record(0, 0,
                [(0.1, 0., True), (0.0, 0., False), (0.0, 0., False)]),
            make_record(0, 0,
                [(0.0, 0., True), (0.5, 0., True), (0., 0., False)]),
        ])
        self.assertEqual(
            model_selection.LeaveOneOutSelectionMethod.run_acc(run_records),
            None
        )

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
import scripts.collect_results
import io
import textwrap

class TestCollectResults(unittest.TestCase):

    def test_format_mean(self):
        self.assertEqual(
            scripts.collect_results.format_mean([0.1, 0.2, 0.3], False),
            '20.0 +/- 4.7')
        self.assertEqual(
            scripts.collect_results.format_mean([0.1, 0.2, 0.3], True),
            '20.0 $\pm$ 4.7')

    def test_print_table_non_latex(self):
        temp_out = io.StringIO()
        sys.stdout = temp_out
        table = [['1', '2'], ['3', '4']]
        scripts.collect_results.print_table(table, 'Header text', ['R1', 'R2'],
            ['C1', 'C2'], colwidth=10, latex=False)
        sys.stdout = sys.__stdout__
        self.assertEqual(
            temp_out.getvalue(),
            textwrap.dedent("""
            -------- Header text
            C1          C2         
            R1          1           2          
            R2          3           4          
            """)
        )

    def test_print_table_latex(self):
        temp_out = io.StringIO()
        sys.stdout = temp_out
        table = [['1', '2'], ['3', '4']]
        scripts.collect_results.print_table(table, 'Header text', ['R1', 'R2'],
            ['C1', 'C2'], colwidth=10, latex=True)
        sys.stdout = sys.__stdout__
        self.assertEqual(
            temp_out.getvalue(),
            textwrap.dedent(r"""
            \begin{center}
            \begin{tabular}{lcc}
            \toprule
            \multicolumn{8}{c}{Header text} \
            \midrule
            \textbf{C1 & \textbf{C2 \\
            \midrule
            R1         & 1          & 2          \\
            R2         & 3          & 4          \\
            \bottomrule
            \end{tabular}
            \end{center}
            """)
        )

    def test_get_grouped_records(self):
        pass # TODO

    def test_print_results_tables(self):
        pass # TODO

    def test_load_records(self):
        pass # TODO

    def test_end_to_end(self):
        """
        Test that collect_results.py's output matches a manually-verified 
        ground-truth when run on a given directory of test sweep data.

        If you make any changes to the output of collect_results.py, you'll need
        to update the ground-truth and manually verify that it's still
        correct. The command used to update the ground-truth is:

        python -m scripts.collect_results --input_dir=misc/test_sweep_data \
            | tee misc/test_sweep_results.txt

        Furthermore, if you make any changes to the data format, you'll also
        need to rerun the test sweep. The command used to run the test sweep is:

        python -m scripts.sweep launch --data_dir=$DATA_DIR \
          --output_dir=misc/test_sweep_data --algorithms ERM \
          --datasets VLCS --steps 1001  --n_hparams 2 --n_trials 2 \
          --command_launcher local
        """
        result = subprocess.run('python -m scripts.collect_results'
            ' --input_dir=misc/test_sweep_data', shell=True,
            stdout=subprocess.PIPE)

        with open('misc/test_sweep_results.txt', 'r') as f:
            ground_truth = f.read()

        self.assertEqual(result.stdout.decode('utf8'), ground_truth)

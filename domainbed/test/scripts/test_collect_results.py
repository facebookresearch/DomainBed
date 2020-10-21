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
from domainbed.scripts import collect_results

from parameterized import parameterized
import io
import textwrap

class TestCollectResults(unittest.TestCase):

    def test_format_mean(self):
        self.assertEqual(
            collect_results.format_mean([0.1, 0.2, 0.3], False)[2],
            '20.0 +/- 4.7')
        self.assertEqual(
            collect_results.format_mean([0.1, 0.2, 0.3], True)[2],
            '20.0 $\pm$ 4.7')

    def test_print_table_non_latex(self):
        temp_out = io.StringIO()
        sys.stdout = temp_out
        table = [['1', '2'], ['3', '4']]
        collect_results.print_table(table, 'Header text', ['R1', 'R2'],
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
        collect_results.print_table(table, 'Header text', ['R1', 'R2'],
            ['C1', 'C2'], colwidth=10, latex=True)
        sys.stdout = sys.__stdout__
        self.assertEqual(
            temp_out.getvalue(),
            textwrap.dedent(r"""
            \begin{center}
            \adjustbox{max width=\textwidth}{%
            \begin{tabular}{lcc}
            \toprule
            \textbf{C1 & \textbf{C2 \\
            \midrule
            R1         & 1          & 2          \\
            R2         & 3          & 4          \\
            \bottomrule
            \end{tabular}}
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

        python -m domainbed.scripts.collect_results --input_dir=domainbed/misc/test_sweep_data \
            | tee domainbed/misc/test_sweep_results.txt

        Furthermore, if you make any changes to the data format, you'll also
        need to rerun the test sweep. The command used to run the test sweep is:

        python -m domainbed.scripts.sweep launch --data_dir=$DATA_DIR \
          --output_dir=domainbed/misc/test_sweep_data --algorithms ERM \
          --datasets VLCS --steps 1001  --n_hparams 2 --n_trials 2 \
          --command_launcher local
        """
        result = subprocess.run('python -m domainbed.scripts.collect_results'
            ' --input_dir=domainbed/misc/test_sweep_data', shell=True,
            stdout=subprocess.PIPE)

        with open('domainbed/misc/test_sweep_results.txt', 'r') as f:
            ground_truth = f.read()

        self.assertEqual(result.stdout.decode('utf8'), ground_truth)

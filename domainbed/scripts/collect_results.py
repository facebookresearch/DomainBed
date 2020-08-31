# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return 'X'
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print('')

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
        print('\multicolumn{8}{c}{'+header_text+'} \\')
        print("\\midrule")
    else:
        print('--------', header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label) + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{center}")

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r['args']['test_envs']:
            group = (r['args']['trial_seed'],
                r['args']['dataset'],
                r['args']['algorithm'],
                test_env)
            result[group].append(r)
    return Q([{'trial_seed': t, 'dataset': d, 'algorithm': a, 'test_env': e,
        'records': Q(r)} for (t,d,a,e),r in result.items()])

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = get_grouped_records(records).map(lambda group:
        { **group, 'sweep_acc': selection_method.sweep_acc(group['records']) }
    ).filter(lambda g: g['sweep_acc'] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select('args.algorithm').unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select('args.dataset').unique().sorted()

    for dataset in dataset_names:
        test_envs = range(datasets.NUM_ENVIRONMENTS[dataset])

        table = [[None for _ in test_envs] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        'dataset, algorithm, test_env',
                        (dataset, algorithm, test_env)
                    ).select('sweep_acc'))
                table[i][j] = format_mean(trial_accs, latex)

        col_labels = [
            'Algorithm', 
            *datasets.get_dataset_class(dataset).ENVIRONMENT_NAMES
        ]
        header_text = (f'Dataset: {dataset}, '
            f'model selection method: {selection_method.name}')
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an 'averages' table

    table = [[None for _ in dataset_names] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals('algorithm, dataset', (algorithm, dataset))
                .group('trial_seed')
                .map(lambda trial_seed, group:
                    group.select('sweep_acc').mean()
                )
            )
            table[i][j] = format_mean(trial_averages, latex)

    col_labels = ['Algorithm', *dataset_names]
    header_text = f'Averages, model selection method: {selection_method.name}'
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, 'results.jsonl')
        try:
            with open(results_path, 'r') as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description='Domain generalization testbed')
    parser.add_argument('--input_dir', type=str, default="")
    parser.add_argument('--latex', action='store_true')
    args = parser.parse_args()

    sys.stdout = misc.Tee(os.path.join(args.input_dir, 'results.txt'), "w")

    records = load_records(args.input_dir)
   
    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\begin{document}")
        print('% Total records:', len(records))
    else:
        print('Total records:', len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.LeaveOneOutSelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")

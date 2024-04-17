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
import matplotlib.pyplot as plt
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings


def plot_data_with_error_bars(data_dict, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    hparams_values, means, errors = zip(*data_dict['value_mean_err'])
    print(data_dict['algorithm'], data_dict['hparams_key'],data_dict['dataset'], data_dict["selection_method"], hparams_values, means, errors)
    # Plotting the graph
    plt.figure(figsize=(8, 5))
    plt.errorbar(hparams_values, means, yerr=errors, fmt='-o', capsize=5)

    plt.xlabel(data_dict['hparams_key'])
    plt.ylabel('Mean Accuracy (%)')
    plt.title(f"{data_dict['algorithm']} on {data_dict['dataset']}")
    # plt.gca().invert_xaxis()
    plt.grid(True)
    pdf_file_path = os.path.join(file_path, "plot.pdf")
    plt.savefig(pdf_file_path, format='pdf')
    
def format_mean(data):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    return mean, err

def print_table(table, header_text, row_labels, col_labels, colwidth=10):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth)

def print_results_tables(records, selection_method):
    """Given all records, print a results table for each dataset."""
    #read hparams and sort (lexicographic order)
    cag_update = Q(records).select("hparams.cag_update").unique().sorted()
    cagrad_c = Q(records).select("hparams.cagrad_c").unique().sorted()
    meta_lr = Q(records).select("hparams.meta_lr").unique().sorted()
    hparams_dict = {"cag_update": cag_update, "cagrad_c": cagrad_c, "meta_lr": meta_lr}
    
    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]
    
    for algorithm in alg_names:
        for dataset in dataset_names:
            for hparams_key, hparams_value in hparams_dict.items():
                value_mean_err=[]
                for hparam in hparams_value:
                    grouped_records = (reporting
                        .get_grouped_records(records, hparams_key, hparam)
                        .map(lambda group: { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) })
                        .filter(lambda g: g["sweep_acc"] is not None)
                        .filter_equals(f"algorithm, dataset", (algorithm, dataset))
                        .group("trial_seed")
                        .map(lambda trial_seed, group:group.select("sweep_acc").mean())
                    )
                    mean, err = format_mean(grouped_records)
                    value_mean_err.append((hparam, mean, err))
                dict = {"algorithm": algorithm, "dataset": dataset, "hparams_key": hparams_key, "value_mean_err": value_mean_err, "selection_method": selection_method.short_name}
                plot_data_with_error_bars(dict,"./domainbed/results/plot/{}/{}/{}/{}/"
                                        .format(algorithm, dataset, selection_method.short_name, hparams_key))
                
       

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./domainbed/results/plot")
    args = parser.parse_args()

    results_file = "results.txt"

    sys.stdout = misc.Tee(os.path.join(results_file), "w")

    records = reporting.load_records(args.input_dir)

    print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        print_results_tables(records, selection_method)


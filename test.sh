#!/bin/bash

CUDA_VISIBLE_DEVICES=2 DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_datasets.TestOverlapDatasets

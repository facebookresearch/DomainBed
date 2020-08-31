# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

DEBUG_DATASETS = ['Debug28', 'Debug224']

def make_minibatches(dataset, batch_size):
    """Test helper to make a minibatches array like train.py"""
    minibatches = []
    for env in dataset:
        X = torch.stack([env[i][0] for i in range(batch_size)]).cuda()
        y = torch.stack([torch.as_tensor(env[i][1])
            for i in range(batch_size)]).cuda()
        minibatches.append((X, y))
    return minibatches

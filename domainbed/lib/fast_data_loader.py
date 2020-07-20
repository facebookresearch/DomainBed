# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                # print(batch)
                yield batch

class FastDataLoader(object):
    INFINITE = 'infinite'
    EPOCH = 'epoch'
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, weights, batch_size, num_workers,
        length=EPOCH):
        super(FastDataLoader, self).__init__()

        if length == self.EPOCH and weights != None:
            raise Exception("Specifying sampling weights with length=EPOCH is "
                "illegal: every datapoint would eventually get sampled exactly "
                "once.")

        if weights == None:
            weights = torch.ones(len(dataset))

        if length == self.INFINITE:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.WeightedRandomSampler(weights,
                    replacement=True,
                    num_samples=batch_size),
                batch_size=batch_size,
                drop_last=True)
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=batch_size,
                drop_last=False
            )

        self.length = length
        self.underlying_length = len(batch_sampler)
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        if self.length == self.INFINITE:
            while True:
                yield next(self._infinite_iterator)
        else:
            for _ in range(len(self)):
                yield next(self._infinite_iterator)

    def __len__(self):
        if self.length == self.INFINITE:
            raise ValueError
        elif self.length == self.EPOCH:
            return self.underlying_length
        else:
            return self.length

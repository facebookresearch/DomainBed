# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Save some representative images from each dataset to disk.
"""
import random
import torch
import argparse
from domainbed import hparams_registry
from domainbed import datasets
import imageio
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets_to_save = ['OfficeHome', 'TerraIncognita', 'DomainNet', 'RotatedMNIST', 'ColoredMNIST', 'SVIRO']

    for dataset_name in tqdm(datasets_to_save):
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)(
            args.data_dir,
            list(range(datasets.num_environments(dataset_name))),
            hparams)
        for env_idx, env in enumerate(tqdm(dataset)):
            for i in tqdm(range(50)):
                idx = random.choice(list(range(len(env))))
                x, y = env[idx]
                while y > 10:
                    idx = random.choice(list(range(len(env))))
                    x, y = env[idx]
                if x.shape[0] == 2:
                    x = torch.cat([x, torch.zeros_like(x)], dim=0)[:3,:,:]
                if x.min() < 0:
                    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
                    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
                    x = (x * std) + mean
                    assert(x.min() >= 0)
                    assert(x.max() <= 1)
                x = (x * 255.99)
                x = x.numpy().astype('uint8').transpose(1,2,0)
                imageio.imwrite(
                    os.path.join(args.output_dir,
                        f'{dataset_name}_env{env_idx}{dataset.ENVIRONMENTS[env_idx]}_{i}_idx{idx}_class{y}.png'),
                    x)

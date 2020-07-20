# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict

import PIL
import torch
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset
from torchvision import transforms
import torchvision.datasets.folder
from torchvision.datasets import CIFAR100, MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import tqdm
import io
import functools

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug2048",
    "Debug28",
    "Debug32",
    "Debug224",
    # Small images
    "RotatedMNIST",
    "ColoredMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
]

NUM_ENVIRONMENTS = {
    # Debug
    "Debug28": 3,
    "Debug32": 3,
    "Debug224": 3,
    "Debug2048": 3,
    # Small images
    "RotatedMNIST": 6,
    "ColoredMNIST": 3,
    # Big images
    "VLCS": 4,
    "PACS": 4,
    "OfficeHome": 4,
    "TerraIncognita": 4,
    "DomainNet": 6,
}


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class MultipleDomainDataset:
    N_STEPS = 10*1000
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8

class Debug(MultipleDomainDataset):
    DATASET_SIZE = 16
    INPUT_SHAPE = None # Subclasses should override
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.environments = [0, 1, 2]
        self.datasets = []
        for _ in range(len(self.environments)):
            self.datasets.append(
                TensorDataset(
                    torch.randn(self.DATASET_SIZE, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (self.DATASET_SIZE,))
                )
            )
    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENT_NAMES = ['0', '1', '2']

class Debug32(Debug):
    INPUT_SHAPE = (3, 32, 32)

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)

class Debug2048(Debug):
    INPUT_SHAPE = (2048,)



class MultipleEnvironmentMNIST(MultipleDomainDataset):
    N_WORKERS = 1

    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=PIL.Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        self.environments = [f.name for f in os.scandir(root) if f.is_dir()]
        self.environments = sorted(self.environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(self.environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class VLCS(MultipleEnvironmentImageFolder):
    N_STEPS = 4000
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, True, hparams)

class PACS(MultipleEnvironmentImageFolder):
    N_STEPS = 4000
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, True, hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 1000
    ENVIRONMENT_NAMES = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, True, hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    N_STEPS = 4000
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, True, hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    N_STEPS = 4000
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, True, hparams)

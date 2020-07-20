DomainBed
=====================================

A PyTorch suite of benchmark dataset and algorithm implementations for domain generalization, as introduced in ["In Search of Lost Domain Generalization"](https://arxiv.org/abs/2007.01434).

## Quick start

First, download the datasets:

``python -m scripts.download --data_dir=/my/datasets/path``

To train a model:

``python -m scripts.train --data_dir=/my/datasets/path --algorithm ERM --dataset RotatedMNIST``

To launch a sweep, write a wrapper for your cluster's job queue in ``command_launchers.py``, and then:

``python -m scripts.sweep launch --data_dir=/my/datasets/path --output_dir=/my/sweep/output/path --command_launcher MyLauncher``

By default this runs a sweep consisting of (all algorithms) x (all datasets) x (3 independent trials) x (20 hparam choices), which at the time of writing is about 50K models in total.
This is excessive for most purposes, so you can pass arguments to make the sweep smaller. For example:

``python -m scripts.sweep launch --data_dir=/my/datasets/path --output_dir=/my/sweep/output/path --command_launcher MyLauncher --algorithms ERM DANN --datasets RotatedMNIST VLCS --n_hparams 5 --n_trials 1``

Some jobs might fail, e.g. if your cluster preempts them. After all the jobs have either succeeded or failed, you can delete the data from jobs which didn't succeed with ``python -m scripts.sweep delete_incomplete`` and then re-launch them by running ``python -m scripts.sweep launch`` again.
In both commands, make sure to specify exactly the same command-line args as you did the first time; this is how the sweep script knows which jobs were launched originally.

Once the sweep is finished, to view the results:

``python -m scripts.collect_results --input_dir=/my/sweep/output/path``

## Implementing new algorithms and datasets

Algorithms are in ``algorithms.py``; datasets are in ``datasets.py``.
Pull requests welcome.

## Running tests

DomainBed includes some unit tests and end-to-end tests. They're not exhaustive, but they're a good sanity-check for your configuration and code.
To run the tests:

``python -m unittest discover``

By default, this only runs tests which don't depend on a dataset directory. To run those tests as well:

``DATA_DIR=/my/datasets/path python -m unittest discover``

## License

This source code is released under the MIT license, included [here](LICENSE). 
